import sys
import cv2
import numpy as np


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


def main(image_name):
	img = cv2.imread(image_name)

	# Convert the image to black and white, then apply Canny edge detection algorithm.
	edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, edges = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	edges = cv2.Canny(edges, 50, 150, apertureSize=3)

	# Find Hough lines, which correspond to edges in the original picture.
	lines = cv2.HoughLines(edges, 2, np.pi / 180, 200)
	num_lines = lines.shape[0]
	if lines is None or num_lines < 20:
		print('Sudoku grid was not detected.')
		return

	# We only care about the horizontal or vertical lines.
	valid_lines = []
	for line in lines:
		for rho, theta in line:
		    if (np.pi / 2 - 0.1) <= theta <= (np.pi / 2 + 0.1) or (3 * np.pi / 2 - 0.1) <= theta <= (3 * np.pi / 2 + 0.1) or \
		    	(np.pi - 0.1) <= theta <= (np.pi + 0.1) or -0.1 <= theta <= 0.1 or (2 * np.pi - 0.1) <= theta <= (2 * np.pi + 0.1):
		    	valid_lines.append((rho, theta))

	print('Number of lines:  {}'.format(num_lines))

	# Finds the intersection points of the lines and plots them.
	for line1 in valid_lines:
		for line2 in valid_lines:
			intersection_point = intersect(line1, line2)
			if intersection_point:
				cv2.circle(img, intersection_point, 1, (255, 0, 0))

	cv2.imwrite('lines.jpg', img)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Usage:  {} <sudoku image>'.format(argv[0]))

	main(sys.argv[1])