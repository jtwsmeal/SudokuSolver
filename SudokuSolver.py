import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans


# Returns None if line1 and line2 do not intersect.
# Returns the intersection point otherwise.
def intersect(line1, line2):
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
		sin_theta1 = 0.01
	cos_theta1 = np.cos(theta1)
	if cos_theta1 == 0:
		cos_theta1 = 0.01
	sin_theta2 = np.sin(theta2)
	if sin_theta2 == 0:
		sin_theta2 = 0.01
	cos_theta2 = np.cos(theta2)
	if cos_theta2 == 0:
		cos_theta2 = 0.01

	# Find the point of intersection.
	x = ((rho2 / sin_theta2) - (rho1 / sin_theta1)) / ((cos_theta2 / sin_theta2) - (cos_theta1 / sin_theta1))
	y = (-cos_theta1 / sin_theta1) * x + (rho1 / sin_theta1)
	return (int(x), int(y))


# Given horizontal lines, returns a set of 10 lines after performing k means.
def get_mean_horz_lines(horz_lines, img_width):
	# Create a list of y intercepts (i.e. where the horizontal lines intersect the line x=img_width / 2)
	x = img_width / 2
	rhos = horz_lines[:,0]
	thetas = horz_lines[:,1]
	sin_thetas = np.sin(thetas)
	sin_thetas[sin_thetas == 0] = 0.01
	y_ints = (1 / sin_thetas) * (rhos - x * np.cos(thetas))
	y_ints = y_ints.reshape(-1, 1)
	num_classes = 10

	# Perform K-means on the x-intercepts.
	kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(y_ints)

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

	mean_rhos = kmeans.cluster_centers_.reshape(10,) * np.sin(mean_thetas) + x * np.cos(mean_thetas)

	# Return the mean lines.
	return np.column_stack((mean_rhos, mean_thetas))


# Given horizontal lines, returns a set of 10 lines after performing k means.
def get_mean_vert_lines(vert_lines, img_height):
	# Create a list of x intercepts (i.e. where the vertical lines intersect the line y=img_height / 2)
	y = img_height / 2
	rhos = vert_lines[:,0]
	thetas = vert_lines[:,1]
	sin_thetas = np.sin(thetas)
	sin_thetas[sin_thetas == 0] = 0.01
	x_ints = -np.tan(thetas) * (y - (rhos / sin_thetas))
	x_ints = x_ints.reshape(-1, 1)
	num_classes = 10

	# Perform K-means on the x-intercepts.
	kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(x_ints)

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

	mean_rhos = y * np.sin(mean_thetas) + kmeans.cluster_centers_.reshape(10,) * np.cos(mean_thetas)

	# Return the mean lines.
	return np.column_stack((mean_rhos, mean_thetas))


def main(image_name):
	img = cv2.imread(image_name)
	img_height, img_width, num_channels = img.shape

	# Convert the image to greyscale (if necessary), then to black and white, then apply Canny edge detection.
	if num_channels == 1:
		edges = img
	else:
		edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	_, edges = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	edges = cv2.Canny(edges, 50, 150, apertureSize=3)

	# Find Hough lines, which correspond to edges in the original picture.
	lines = cv2.HoughLines(edges, 2, np.pi / 180, 200)
	if lines is None or lines.shape[0] < 20:
		print('Sudoku grid was not detected.')
		return

	# Change array shape from (num_lines, 1, 2) to (num_lines, 2).
	lines = np.squeeze(lines)
	num_lines = lines.shape[0]

	# We only care about the horizontal or vertical lines.
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

	num_valid_lines = valid_horz_lines.shape[0] + valid_vert_lines.shape[0]
	if num_valid_lines < 20:
		print('Sudoku grid was not detected.')
		return

	print('Number of valid lines:  {}'.format(num_valid_lines))

	mean_horiz_lines = get_mean_horz_lines(valid_horz_lines, img_width)
	mean_vert_lines = get_mean_vert_lines(valid_vert_lines, img_height)

	for rho, theta in np.vstack((mean_vert_lines, mean_horiz_lines)):
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 - 1000 * b)
		y1 = int(y0 + 1000 * a)
		x2 = int(x0 + 1000 * b)
		y2 = int(y0 - 1000 * a)

		cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

	# # Finds the intersection points of the lines and plots them.
	# intersection_points = []
	# for line1 in valid_lines:
	# 	for line2 in valid_lines:
	# 		intersection_point = intersect(line1, line2)
	# 		if intersection_point:
	# 			intersection_points.append(intersection_point)
	# 			# cv2.circle(img, intersection_point, 1, (255, 0, 0))

	# cluster_centers = do_kmeans(intersection_points)
	# for centre in cluster_centers:
	# 	cv2.circle(img, (centre[0], centre[1]), 3, (255, 0, 0))


	cv2.imwrite('lines.jpg', img)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Usage:  {} <sudoku image>'.format(argv[0]))
		exit(1)

	main(sys.argv[1])
