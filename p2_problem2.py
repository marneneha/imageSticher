"""
Author: Mudit Singal
UID: 119262689
Dir ID: msingal

Submission: Project 2, Problem 2

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

# Function to scale the window size for properly displaying the images
def show_image_reshaped(img, title, width, height):
	cv2.namedWindow(title, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(title, width, height)
	cv2.imshow(title, img)
	return

# Function to convert image to grayscale, detect features and return key points and descriptors from image and specified feature detector
def get_features_in_img(img, detector):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detected_keypts, descriptors = detector.detectAndCompute(gray_img, mask=None)
	return detected_keypts, descriptors, gray_img

# CV2 Matcher parameters:
# normType = cv2.NORM_L2 (good for SIFT, SURF), cv2.NORM_L1 | cv2.NORM_HAMMING for ORB, BRIEF, BRISK
# crossCheck = False (default), set as True to if matching descriptors in set A and set B are best for both sets
# BFMatcher.match() returns 1 only and BFMatcher.knnMatch() returns n 


# Construct the A matrix as per the given formula in slides from 2 sets of 4 points from 2 images feature matches
def make_A_matrix(image1_pts, image2_pts):
	A = np.zeros(shape=(8,9), dtype=np.float32)
	for i in range(0,7,2):
		x1 = image2_pts[i//2, 0]
		y1 = image2_pts[i//2, 1]

		x1_dash = image1_pts[i//2, 0]
		y1_dash = image1_pts[i//2, 1]
		A[i] = np.array([x1, y1, 1, 0, 0, 0, -x1_dash*x1, -x1_dash*y1, -x1_dash])
		A[i+1] = np.array([0, 0, 0, x1, y1, 1, -y1_dash*x1, -y1_dash*y1, -y1_dash])

	# print(A)
	return A


# Function to calculate the H matrix as by finding the least eigen vector of A_transpose * A, scaling it by the h22 term 
# and then rearranging into a 3x3 matrix
def calc_H_matrix(A):
	sq_A = np.matmul(A.T, A)
	eig_vals, eig_vecs = np.linalg.eig(sq_A)
	smallest_eig_vec = eig_vecs[:, np.argmin(eig_vals)]
	smallest_eig_vec = smallest_eig_vec / smallest_eig_vec[-1]
	H = smallest_eig_vec.reshape((3,3))

	return H

# Function to join the 2 images using wrapPerspective and adding the first image to the left of the second one
def join_images(img1, img2, H):
	final_height = max(img1.shape[0], img2.shape[0])
	final_width = img1.shape[1] + img2.shape[1]
	stitched_img = cv2.warpPerspective(img2, H, (final_width,final_height))
	stitched_img[:img1.shape[0], :img1.shape[1]] = img1

	return stitched_img, final_height, final_width

# FUnction to find H matrix using RANSAC. 
# For 'iterations' number of iterations, we take 4 random points and find the number of inliers
def find_H_RANSAC(pts1, pts2, threshold = 100, iterations = 200):

	# Take 4 random points from the detected features list
	indices = np.random.randint(pts1.shape[0], size=4)
	img1_selected_pts = pts1[indices, :]
	img2_selected_pts = pts2[indices, :]

	# Calculating the initial A and H matrices
	A = make_A_matrix(img1_selected_pts, img2_selected_pts) 	# A matrix
	H = calc_H_matrix(A)
	best_H = H

	n_inliers = 0
	n_inliers_best = 0
	best_inliers = []

	for i in range(iterations):
		n_inliers = 0
		inliers_p1 = []
		inliers_p2 = []

		# Find inliers by comparing x' and x for each point in array of matches
		for j in range(pts1.shape[0]):
			pt1_j = pts1[j,:]
			pt2_j = pts2[j,:]

			# Estimate of points in image 2 in perspective of image 1, by using the H matrix
			est_pt1 = np.matmul(H, [pt2_j[0], pt2_j[1], 1])
			est_pt1 = (est_pt1 / est_pt1[2])[:2]

			# Find the euclidean distance between the estimated point and the actual point
			dist = np.linalg.norm(pt1_j - est_pt1)


			# Only take the point as inlier if its distance is less than a threshold
			if dist < threshold:
				inliers_p1.append(pt1_j)
				inliers_p2.append(pt2_j)
				n_inliers += 1
		
		# If the new H is better than previous best, set best H as current H and also store the best inlier points
		if n_inliers > n_inliers_best:
			n_inliers_best = n_inliers
			best_H = H
			best_inliers_p1 = inliers_p1
			best_inliers_p2 = inliers_p2

		# inliers_p1.clear()
		# inliers_p2.clear()

		# Take new 4 random points and calculate A and H matrices from them
		indices = np.random.randint(pts1.shape[0], size=4)
		img1_selected_pts = pts1[indices, :]
		img2_selected_pts = pts2[indices, :]
		# Calculating the A, H and P matrices
		A = make_A_matrix(img1_selected_pts, img2_selected_pts) 	# A matrix
		H = calc_H_matrix(A)


	# Using all the inliers, find the best H to get a more accurate estimate, not necessary but this gives less variation in results
	A = np.zeros(shape=( 2*n_inliers_best ,9), dtype=np.float32)
	print(n_inliers_best)
	print(len(best_inliers_p2))
	print(len(best_inliers_p1))
	print(A.shape)
	for i in range(0, 2*n_inliers_best - 1, 2):
		x1 = best_inliers_p2[i//2][0]
		y1 = best_inliers_p2[i//2][1]

		x1_dash = best_inliers_p1[i//2][0]
		y1_dash = best_inliers_p1[i//2][1]
		A[i] = np.array([x1, y1, 1, 0, 0, 0, -x1_dash*x1, -x1_dash*y1, -x1_dash])
		A[i+1] = np.array([0, 0, 0, x1, y1, 1, -y1_dash*x1, -y1_dash*y1, -y1_dash])

	best_H = calc_H_matrix(A)

	return best_H
			

# Function to compute the homography matrix between 2 images
def compute_homography(detected_kps1, detected_kps2, matches):
	pt1_arr = np.empty(shape=(len(matches), 2))
	pt2_arr = np.empty(shape=(len(matches), 2))
	print(pt1_arr.shape)
	i = 0

	# Finding point in images corresponding to the matching keypoints
	for match in matches:
		pt1_arr[i] = detected_kps1[match.queryIdx].pt
		pt2_arr[i] = detected_kps2[match.trainIdx].pt
		i += 1


	H = find_H_RANSAC(pt1_arr, pt2_arr, threshold = 4, iterations=1000)		# Homography matrix
	print(H)
	
	return H


# Reading the images
curr_pwd = os.getcwd()
img1 = cv2.imread(curr_pwd + '/problem_2_images/image_1.jpg')
img2 = cv2.imread(curr_pwd + '/problem_2_images/image_2.jpg')
img3 = cv2.imread(curr_pwd + '/problem_2_images/image_3.jpg')
img4 = cv2.imread(curr_pwd + '/problem_2_images/image_4.jpg')

# Making copies to avoid corruption of images
orig_img1 = img1.copy()
orig_img2 = img2.copy()
orig_img3 = img3.copy()
orig_img4 = img4.copy()

# Scaling images to avoid openCV error
scale_fraction = 0.9
width = int(img1.shape[1] * scale_fraction)
height = int(img1.shape[0] * scale_fraction)
dim = (width, height)

img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
img3 = cv2.resize(img3, dim, interpolation = cv2.INTER_AREA)
img4 = cv2.resize(img4, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',img1.shape)


# ORB detector
orb = cv2.ORB_create(1000)

# Detect features in all 4 images using orb detector
detected_kps1, des1, gray1 = get_features_in_img(img1, detector=orb)
detected_kps2, des2, gray2 = get_features_in_img(img2, detector=orb)
detected_kps3, des3, gray3 = get_features_in_img(img3, detector=orb)
detected_kps4, des4, gray4 = get_features_in_img(img4, detector=orb)

# Drawing keypoints in images for visualization (not visualized currently)
detected_kps_img1 = cv2.drawKeypoints(gray1, detected_kps1, img1)
detected_kps_img2 = cv2.drawKeypoints(gray2, detected_kps2, img2)
detected_kps_img3 = cv2.drawKeypoints(gray3, detected_kps3, img3)
detected_kps_img4 = cv2.drawKeypoints(gray4, detected_kps4, img4)

# Uncomment to view detected keypoints
# show_image_reshaped(img=detected_kps_img1, title="Detected1", width=756, height=1008)
# show_image_reshaped(img=detected_kps_img2, title="Detected2", width=756, height=1008)
# show_image_reshaped(img=detected_kps_img3, title="Detected3", width=756, height=1008)
# show_image_reshaped(img=detected_kps_img4, title="Detected4", width=756, height=1008)

# Creating a brute force matcher
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Find matches between image 1 and image 2 and sort them using distance
matches_1_2 = bf_matcher.match(des1, des2)
matches_1_2 = sorted(matches_1_2, key = lambda x:x.distance)

# Find matches between image 3 and image 4 and sort them using distance
matches_3_4 = bf_matcher.match(des3, des4)
matches_3_4 = sorted(matches_3_4, key = lambda x:x.distance)

# Visualize the matches
img_match = cv2.drawMatches(gray1, detected_kps1, gray2, detected_kps1, matches_1_2[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_match2 = cv2.drawMatches(gray3, detected_kps3, gray4, detected_kps4, matches_3_4[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Find homography between image 1 and image 2
H = compute_homography(detected_kps1, detected_kps2, matches_1_2[:500])

# Join the 2 images, crop it and display it
s_image1, w, h = join_images(orig_img1, orig_img2, H)
mid = s_image1.shape[0]//2
for px in range(s_image1.shape[1]):
	if np.sum(s_image1[mid, px]) == 0:
		col_zero = px
		break

s_image1 = s_image1[:, :col_zero, :]
s_image1_cp = s_image1.copy()
# show_image_reshaped(img=s_image1, title="Stitched Img 1", width=col_zero, height=h)


# Find homography between image 3 and image 4
H = compute_homography(detected_kps3, detected_kps4, matches_3_4[:500])

# Join the 2 images
s_image2, w, h = join_images(orig_img3, orig_img4, H)

# Cropping the excess portion
s_image2 = s_image2[:, :-190, :]

s_image2_cp = s_image2.copy()

# Detect the features between stitches of image 1, 2 and stitches of image 3,4
detected_kps5, des5, gray5 = get_features_in_img(s_image1, detector=orb)
detected_kps6, des6, gray6 = get_features_in_img(s_image2, detector=orb)

# Find matches for above features and keep them in a sorted list
matches_s1_s2 = bf_matcher.match(des5, des6)
matches_s1_s2 = sorted(matches_s1_s2, key = lambda x:x.distance)

# Visualize the final matches
img_match3 = cv2.drawMatches(gray5, detected_kps5, gray6, detected_kps6, matches_s1_s2[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Compute Homography matrix between stitch 1 (image1+image2) and stitch 2 (image3+image4)
H = compute_homography(detected_kps5, detected_kps6, matches_s1_s2[:500])

# Join the 2 final images
final_s_img, w, h = join_images(s_image1_cp, s_image2_cp, H)

# Visualize different matches
show_image_reshaped(img=img_match, title="Matches", width=1036, height=1008)
show_image_reshaped(img=img_match2, title="Matches2", width=1036, height=1008)
show_image_reshaped(img=img_match3, title="Matches3", width=1036, height=1008)


# Visualize the final stitched panorama :)
show_image_reshaped(img=final_s_img, title="Stitched Img", width=final_s_img.shape[1], height=final_s_img.shape[0])

# Press any key to exit once the images have been visualized
cv2.waitKey(0)
cv2.destroyAllWindows()