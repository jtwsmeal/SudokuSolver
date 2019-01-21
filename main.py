import os
import sys
import cv2
import numpy as np
import square_detection
from sklearn.cluster import KMeans
from skimage import segmentation
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras import models

import solve_sudoku

# To prevent a CPU warning.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Resizes image given a minimum height/width.
def resize_img(img, threshold):
	img_height, img_width, _ = img.shape
	if img_height > threshold and img_width > threshold:
		return
	else:
		img = cv2.resize(img, (max(img_height, threshold), max(img_width, threshold)), interpolation=cv2.INTER_AREA)


# Applies edge detection to the image and dilates it.
def preprocess_img(img, kernel_size, dilation_iters):
	kernel = np.ones(kernel_size, np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.Canny(img, 50, 150, apertureSize=3)
	img = cv2.dilate(img, kernel, iterations=dilation_iters)
	return img


# Given an array of points on squares, returns an array of the areas of these squares.
def get_square_areas(squares):
	v1 = squares[:, 1] - squares[:, 0]
	v2 = squares[:, 2] - squares[:, 1]
	len_v1 = np.linalg.norm(v1, axis=1)
	len_v2 = np.linalg.norm(v2, axis=1)
	v1_unit = v1 / len_v1[:, np.newaxis]
	v2_unit = v2 / len_v2[:, np.newaxis]
	theta = np.arccos(np.clip(np.einsum('ij,ij->i', v1_unit, v2_unit), -1.0, 1.0))
	return len_v1 * len_v2 * np.sin(theta)


# Removes squares that are have an area much larger or much smaller than the median square area.
def remove_odd_sized_squares(squares):
	# Find the average areas of the squares and remove a square if its area is an outlier.
	square_areas = get_square_areas(squares)
	median_area = np.median(square_areas)
	square_removal_indices = np.where(
		np.logical_or(
			(square_areas - median_area) / (square_areas + median_area) < -0.25,
			(square_areas - median_area) / (square_areas + median_area) > 0.5
		)
	)

	# Remove the squares that have outlier areas.
	return np.delete(squares, square_removal_indices[0], axis=0)


# Removes squares that are rotated differently than the others.
# Returns the updated squares and the mean thetas as computed in k-means.
def remove_crooked_squares(squares):
	num_squares = squares.shape[0]
	thetas = np.array([])

	for i in range(4):
		numerator = squares[:, (i+1) % 4, 1] - squares[:, i, 1]
		denominator = (squares[:, (i+1) % 4, 0] - squares[:, i, 0]).astype(float)

		# Prevent a denominator of 0.
		denominator[denominator == 0] = 0.001
		thetas = np.append(thetas, np.arctan(numerator / denominator))

	# Translate angle -> 2D point, perform k-means, then eliminate the crooked squares.
	# The purpose of this translation is so that angles near 0 and angles near pi are in the same cluster.
	pts_on_unit_circle = np.c_[np.cos(2 * thetas), np.sin(2 * thetas)]
	kmeans = KMeans(n_clusters=2, random_state=0).fit(pts_on_unit_circle)
	mean_pts_on_unit_circle = kmeans.cluster_centers_
	distances_from_mean = np.linalg.norm(mean_pts_on_unit_circle[kmeans.labels_] - pts_on_unit_circle, axis=1)

	# We know that a square is crooked if its distance from its mean angle is past a threshold.
	locations_of_crooked_squares = np.where(distances_from_mean > 0.4)[0]

	if locations_of_crooked_squares.shape[0] != 0:
		# Perform k-means again to obtain the mean points on the unit circle based
		# on the valid thetas (which don't belong to a square being removed).
		thetas = np.delete(thetas, locations_of_crooked_squares, axis=0)
		pts_on_unit_circle = np.c_[np.cos(2 * thetas), np.sin(2 * thetas)]
		kmeans = KMeans(n_clusters=2, random_state=0).fit(pts_on_unit_circle)
		mean_pts_on_unit_circle = kmeans.cluster_centers_

	mean_thetas = np.arctan2(mean_pts_on_unit_circle[:, 1], mean_pts_on_unit_circle[:, 0]) / 2.0

	# Note that we divide the indices by 4 since we analyze the crookedness of
	# each square's lines, which there are four of.
	return np.delete(squares, locations_of_crooked_squares / 4, axis=0), mean_thetas


def remove_oddly_located_squares(squares):
	# TODO:  Perform k-medians on the lines of the squares, eliminate outlier squares.
	return squares


# Rotates the image according to the mean angles of the grid cells, given by mean_thetas.
def rotate_img(img, img_width, img_height, mean_thetas):
	img_centre = (img_width / 2, img_height / 2)

	# Rotate the image in the direction that requires the least amount of rotation.
	rotation_angle = np.min(np.abs(mean_thetas))
	rotation_matrix = cv2.getRotationMatrix2D(center=img_centre, angle=rotation_angle * 180 / np.pi, scale=1)
	new_img_width = int((img_height * np.abs(rotation_matrix[0, 1])) + (img_width * np.abs(rotation_matrix[0, 0])))
	new_img_height = int((img_height * np.abs(rotation_matrix[0, 0])) + (img_width * np.abs(rotation_matrix[0, 1])))
	rotation_matrix[0, 2] += (new_img_width / 2) - img_centre[0]
	rotation_matrix[1, 2] += (new_img_height / 2) - img_centre[1]
	img = cv2.warpAffine(img, rotation_matrix, (new_img_width, new_img_height))
	return img, rotation_matrix


# Returns a tuple of four pairs of points, where each pair of points defines an edge of the Sudoku grid.
# Note that we determine the second point of each edge by shifting the first point in the appropriate
# direction by 1 unit, so that the points are on the same axis and the lines are flat.
def get_grid_edges(squares):

	left_pt1 = squares[np.unravel_index(np.argmin(squares[:, :, 0]), squares[:, :, 0].shape)].astype(float)
	left_pt2 = left_pt1.copy()
	left_pt2[1] += 1
	left_pts = [left_pt1, left_pt2]

	right_pt1 = squares[np.unravel_index(np.argmax(squares[:, :, 0]), squares[:, :, 0].shape)].astype(float)
	right_pt2 = right_pt1.copy()
	right_pt2[1] += 1
	right_pts = [right_pt1, right_pt2]

	top_pt1 = squares[np.unravel_index(np.argmin(squares[:, :, 1]), squares[:, :, 1].shape)].astype(float)
	top_pt2 = top_pt1.copy()
	top_pt2[0] += 1
	top_pts = [top_pt1, top_pt2]

	bottom_pt1 = squares[np.unravel_index(np.argmax(squares[:, :, 1]), squares[:, :, 1].shape)].astype(float)
	bottom_pt2 = bottom_pt1.copy()
	bottom_pt2[0] += 1
	bottom_pts = [bottom_pt1, bottom_pt2]

	return left_pts, right_pts, top_pts, bottom_pts


def preprocess_clear_img(clear_img):
	kernel = np.ones((3, 3), np.uint8)
	clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2GRAY)
	clear_img = cv2.adaptiveThreshold(clear_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	clear_img = cv2.morphologyEx(clear_img, cv2.MORPH_CLOSE, kernel, iterations=1)
	return clear_img


# Given a set of lines, returns the vertical and horizontal lines, based on an angle threshold.
def get_valid_lines(lines):
	thetas = lines.T[1]
	horiz_cond = np.logical_or(
		np.logical_and((np.pi / 2 - 0.1) <= thetas, thetas <= (np.pi / 2 + 0.1)),
		np.logical_and((3 * np.pi / 2 - 0.1) <= thetas, thetas <= (3 * np.pi / 2 + 0.1))
	)
	valid_horz_lines = lines[np.where(horiz_cond)]

	vert_cond = np.logical_or(
		np.logical_and((np.pi - 0.1) <= thetas, thetas <= (np.pi + 0.1)),
		thetas <= 0.1,
		(2 * np.pi - 0.1) <= thetas
	)
	valid_vert_lines = lines[np.where(vert_cond)]

	return valid_horz_lines, valid_vert_lines


# Given horizontal lines, returns a set of 10 lines after performing k means.
def get_mean_horz_lines(horz_lines, img_width):
	# Create a list of y intercepts (i.e. where the horizontal lines intersect the line x=img_width / 2)
	x = img_width / 2
	rhos = horz_lines[:,0]
	thetas = horz_lines[:,1]
	sin_thetas = np.sin(thetas)
	sin_thetas[sin_thetas == 0] = 0.001
	y_ints = (1 / sin_thetas) * (rhos - x * np.cos(thetas))
	num_classes = 10

	# Perform K-means on the x-intercepts.
	kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(y_ints.reshape(-1, 1))

	# Sort the labels and sort the thetas based on the order of the labels.
	sorted_label_indices = kmeans.labels_.argsort()
	sorted_labels = kmeans.labels_[sorted_label_indices]
	sorted_thetas = thetas[sorted_label_indices]

	mean_thetas = np.zeros(num_classes)
	num_labels_passed = 0
	for k in range(num_classes):
		num_lines_of_class = np.extract(sorted_labels == k, sorted_labels).shape[0]
		mean_thetas[k] = np.mean(sorted_thetas[num_labels_passed:num_labels_passed + num_lines_of_class])
		num_labels_passed += num_lines_of_class

	centres = kmeans.cluster_centers_.reshape(10,)
	mean_rhos = centres * np.sin(mean_thetas) + x * np.cos(mean_thetas)

	centre_indices = centres.reshape(10,).argsort()
	sorted_mean_rhos = mean_rhos[centre_indices]
	sorted_mean_thetas = mean_thetas[centre_indices]

	# Return the mean lines.
	return np.column_stack((sorted_mean_rhos, sorted_mean_thetas))


# Given horizontal lines, returns a set of 10 lines after performing k means.
def get_mean_vert_lines(vert_lines, img_height):
	# Create a list of x intercepts (i.e. where the vertical lines intersect the line y=img_height / 2)
	y = img_height / 2
	rhos = vert_lines[:,0]
	thetas = vert_lines[:,1]
	sin_thetas = np.sin(thetas)
	sin_thetas[sin_thetas == 0] = 0.001
	x_ints = -np.tan(thetas) * (y - (rhos / sin_thetas))
	num_classes = 10

	# Perform K-means on the x-intercepts.
	kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(x_ints.reshape(-1, 1))

	# Sort the labels and sort the thetas based on the order of the labels.
	sorted_label_indices = kmeans.labels_.argsort()
	sorted_labels = kmeans.labels_[sorted_label_indices]
	sorted_thetas = thetas[sorted_label_indices]

	# If the angle is above pi / 2, it is near pi, so we temporarily bring it down near zero so it is near the other angles in its cluster.
	sorted_thetas[sorted_thetas > np.pi / 2] -= np.pi
	mean_thetas = np.zeros(num_classes)
	num_labels_passed = 0
	for k in range(num_classes):
		num_lines_of_class = np.extract(sorted_labels == k, sorted_labels).shape[0]
		mean_thetas[k] = np.mean(sorted_thetas[num_labels_passed:num_labels_passed + num_lines_of_class])
		num_labels_passed += num_lines_of_class

	# Make all angles positive.
	mean_thetas[mean_thetas < 0] += np.pi

	centres = kmeans.cluster_centers_.reshape(10,)
	mean_rhos = y * np.sin(mean_thetas) + centres * np.cos(mean_thetas)

	centre_indices = centres.argsort()
	sorted_mean_rhos = mean_rhos[centre_indices]
	sorted_mean_thetas = mean_thetas[centre_indices]

	# Return the mean lines.
	return np.column_stack((sorted_mean_rhos, sorted_mean_thetas))


# Draws the given lines on the given image.  Used for debugging purposes.
def draw_lines_on_img(img, lines, colour, thickness):
	for rho, theta in lines:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 - 1000 * b)
		y1 = int(y0 + 1000 * a)
		x2 = int(x0 + 1000 * b)
		y2 = int(y0 - 1000 * a)
		cv2.line(img, (x1, y1), (x2, y2), colour, thickness)


# Note that line1 and line2 must be numpy arrays of the form
# [rho, theta], where rho = x cos(theta) + y cos(theta).
#
# Returns None if line1 and line2 do not intersect.
# Returns the intersection point otherwise, as a list of scalars
# (since we may need to modify them).
# TODO:  Can this function and the next one be done with np.linalg.solve?
def intersect_rho_theta(line1, line2):
	rho1 = line1[0]
	theta1 = line1[1]
	rho2 = line2[0]
	theta2 = line2[1]

	# Check for no intersection between the lines.
	if theta1 == theta2:
		return

	# Ensure that 0 isn't a denominator.
	sin_theta1 = np.sin(theta1)
	if sin_theta1 == 0:
		sin_theta1 = 0.001
	cos_theta1 = np.cos(theta1)
	if cos_theta1 == 0:
		cos_theta1 = 0.001
	sin_theta2 = np.sin(theta2)
	if sin_theta2 == 0:
		sin_theta2 = 0.001
	cos_theta2 = np.cos(theta2)
	if cos_theta2 == 0:
		cos_theta2 = 0.001

	# Find the point of intersection.
	x = ((rho2 / sin_theta2) - (rho1 / sin_theta1)) / ((cos_theta2 / sin_theta2) - (cos_theta1 / sin_theta1))
	y = (-cos_theta1 / sin_theta1) * x + (rho1 / sin_theta1)

	return [int(x), int(y)]


# Note that line_1 and line_2 are tuples containing two points on the lines.
#
# Returns None if line1 and line2 do not intersect.
# Returns the intersection point otherwise, as a list of scalars
# (since we may need to modify them).
def intersect_endpoints(line1, line2):
	(x1, y1), (x2, y2) = line1
	(x3, y3), (x4, y4) = line2

	if x1 == x2 and x3 == x4:
		return
	elif x1 == x2:
		slope_l2 = (y4 - y3) / (x4 - x3)
		return [x1, slope_l2 * (x1 - x3) + y3]
	elif x3 == x4:
		slope_l1 = (y2 - y1) / (x2 - x1)
		return [x3, slope_l1 * (x3 - x1) + y1]
	elif (y2 - y1) / (x2 - x1) == (y4 - y3) / (x4 - x3):
		# The slopes are the same, so the lines don't intersect.
		return

	slope_l1 = (y2 - y1) / (x2 - x1)
	slope_l2 = (y4 - y3) / (x4 - x3)
	x = (y3 - y1 - slope_l2 * x3 + slope_l1 * x1) / (slope_l1 - slope_l2)
	y = slope_l2 * (x - x3) + y3

	return [x, y]


# Uses a homography to take a portion of the given image, between the 4 given lines.
# is_cell is True iff you want certain operations to occur on the resulting image, such
# as the clearing of the image's borders.  rho_theta is True if the left, right, top, and
# bottom lines are of the form [rho, theta], and False if they are a collection of endpoints.
def crop_img(src_img, left, right, top, bottom, dest_width, dest_height, is_cell, rho_theta):

	# Find the corners of the grid cell.
	if rho_theta:
		# Lines are NumPy arrays containing the rho and theta parameters.
		top_left = intersect_rho_theta(left, top)
		bottom_left = intersect_rho_theta(left, bottom)
		top_right = intersect_rho_theta(right, top)
		bottom_right = intersect_rho_theta(right, bottom)
	else:
		# Lines are of the form (x1, y1, x2, y2), where (x1, y1) and (x2, y2) are the endpoints.
		top_left = intersect_endpoints(left, top)
		bottom_left = intersect_endpoints(left, bottom)
		top_right = intersect_endpoints(right, top)
		bottom_right = intersect_endpoints(right, bottom)

	if top_left is None or bottom_left is None or top_right is None or bottom_right is None:
		print('Lines do not intersect!')
		return

	# Perform a homography transformation to isolate the grid cell in a 28 * 28 array.
	if is_cell:
		src_pts = np.array([top_left, bottom_left, top_right, bottom_right])
	else:
		top_left[0] -= 5
		top_left[1] -= 5
		bottom_right[0] += 5
		bottom_right[1] += 5
		bottom_left[0] -= 5
		bottom_left[1] += 5
		top_right[0] += 5
		top_right[1] -= 5
		src_pts = np.array([top_left, bottom_left, top_right, bottom_right])
	dest_pts = np.array([(0, 0), (0, dest_height), (dest_width, 0), (dest_width, dest_height)])
	homo, _ = cv2.findHomography(src_pts, dest_pts)

	dest_img = cv2.warpPerspective(src_img, homo, (dest_width, dest_height))
	if is_cell:
		_, dest_img = cv2.threshold(dest_img, 128, 255, cv2.THRESH_BINARY_INV)

		# Clear the first three pixels of the image borders, and anything directly attached to them.
		for i in range(3):
			dest_img[i:28-i, i:28-i] = segmentation.clear_border(dest_img[i:28-i, i:28-i])

	return dest_img


def get_sudoku_grid(mean_vert_lines, mean_horiz_lines, src_img):
	# Note that a zero will represent a blank space.
	sudoku_grid = np.zeros(81).reshape((9, 9))
	img_len = 28
	digit_imgs = []
	digit_locations = []

	# Fill in the sudoku grid properly.
	for i in range(9):
		for j in range(9):
			left = mean_vert_lines[i]
			right = mean_vert_lines[i+1]
			top = mean_horiz_lines[j]
			bottom = mean_horiz_lines[j+1]

			dest_img = crop_img(src_img, left, right, top, bottom, img_len, img_len, is_cell=True, rho_theta=True)

			# Uncomment the next line to save an image of each cell.
			# cv2.imwrite('images/grid_cells/cell_' + str(j + 1) + str(i + 1) + '.jpg', dest_img)

			# Only make a prediction for the image if it is not a blank space.
			# To distinguish a number from some noise in a cell, we sum the image's
			# values and check whether they are past a threshold.
			if np.sum(dest_img) > 10000:
				digit_imgs.append(dest_img)
				digit_locations.append((i, j))

	digit_imgs = np.array(digit_imgs)[:, :, :, np.newaxis]

	# Predict the digit.
	with tf.Graph().as_default():
		with tf.Session() as session:
			model = models.load_model('digits.h5')
			backend.set_session(session)

			# Note:  You can use model.predict_proba() to obtain the class probabilities.
			predictions = model.predict(digit_imgs)
			pred_nums = predictions.argmax(axis=1)
			for pred_index, (i, j) in enumerate(digit_locations):
				sudoku_grid[i][j] = pred_nums[pred_index]

	# Take the transpose of the grid.
	sudoku_grid = sudoku_grid.T
	return sudoku_grid


def main(img_name):

	# The method used to isolate the sudoku grid is to gather a collection of (not necessarily all) of
	# the squares from the sudoku grid and infer the location of the grid based on this information.
	orig_img = cv2.imread(img_name)
	resize_img(orig_img, 350)
	img_height, img_width, _ = orig_img.shape

	# img will be used to extract the sudoku grid from the original image.
	img = orig_img.copy()
	img = preprocess_img(img, kernel_size=(3, 3), dilation_iters=2)

	# clear_img will be used to extract the content from each cell.
	clear_img = orig_img.copy()

	# Obtain a list of contours on the image of contours that are squares.
	squares = square_detection.find_squares(img)
	if squares.shape[0] == 0:
		print('Sudoku grid was not detected.')
		return

	# Get rid of outlier squares based on area.
	squares = remove_odd_sized_squares(squares)

	# Get rid of outlier squares based on rotation.
	squares, mean_thetas = remove_crooked_squares(squares)

	# Get rid of outlier squares based on location.
	squares = remove_oddly_located_squares(squares)

	# Find image's angle and rotate it.
	img, rotation_matrix = rotate_img(img, img_width, img_height, mean_thetas)
	clear_img, _ = rotate_img(clear_img, img_width, img_height, mean_thetas)

	# Rotate the squares.  Note that we must add in an entry of 1 to each pair of points in order to allow the shear to occur.
	num_squares = squares.shape[0]
	squares = np.concatenate([squares, np.ones(num_squares * 4).reshape(num_squares, 4, 1)], axis=2)
	squares = np.einsum('ij,klj->kli', rotation_matrix, squares).astype(int)

	# In order to obtain the lines of the grid, we search for the outer lines in squares.
	# Note that since the square array's points are in a certain order, knowing the location
	# in the squares tensor of the leftmost point, for example, allows us to figure out the
	# other point in the leftmost line.
	dest_len = 400

	left_pts, right_pts, top_pts, bottom_pts = get_grid_edges(squares)

	img = crop_img(img, left_pts, right_pts, top_pts, bottom_pts, dest_len, dest_len, is_cell=False, rho_theta=False)
	clear_img = crop_img(clear_img, left_pts, right_pts, top_pts, bottom_pts, dest_len, dest_len, is_cell=False, rho_theta=False)

	# Preprocess clear_img, which is the original image but rotated and cropped.
	clear_img = preprocess_clear_img(clear_img)

	# Find Hough lines, which correspond to edges in the original picture.
	lines = cv2.HoughLines(img, 1, np.pi / 180, 350).squeeze()
	if lines is None or lines.shape[0] < 20:
		print('Sudoku grid was not detected.')
		return

	# Obtain valid lines (eliminate noisy lines through clustering).
	valid_horz_lines, valid_vert_lines = get_valid_lines(lines)
	num_valid_lines = valid_horz_lines.shape[0] + valid_vert_lines.shape[0]
	if num_valid_lines < 20:
		print('Sudoku grid was not detected.')
		return

	# Perform k-means to obtain mean lines.
	mean_horiz_lines = get_mean_horz_lines(valid_horz_lines, img_width)
	mean_vert_lines = get_mean_vert_lines(valid_vert_lines, img_height)

	# Obtain and solve the sudoku.
	sudoku_grid = get_sudoku_grid(mean_vert_lines, mean_horiz_lines, clear_img)
	print(sudoku_grid)
	solution = solve_sudoku.solve(sudoku_grid)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Usage:  {} <sudoku image>'.format(argv[0]))
		exit(1)

	main(sys.argv[1])
