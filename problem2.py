import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import math
import copy

# from RemoveBlackEdge import remove_black_edge
def black_edge_remove(img):
    for i in range(img.shape[1]):
        if(np.sum(img[:, i])==0):
            break
    cropped_img = img[:, 0:i-3]
    return cropped_img

# import glob
images = []
# image_paths = glob.glob('problem_2_images/image_1.jpg')
sift = cv2.SIFT_create()
orb = cv2.ORB_create(1000)

img1 = cv2.imread('problem_2_images/image_1.jpg')
img2 = cv2.imread('problem_2_images/image_2.jpg')
img3 = cv2.imread('problem_2_images/image_3.jpg')
img4 = cv2.imread('problem_2_images/image_4.jpg')
img1 = cv2.resize(img1, (0, 0), fx = 0.15, fy = 0.15)
img2 = cv2.resize(img2, (0, 0), fx = 0.15, fy = 0.15)
img3 = cv2.resize(img3, (0, 0), fx = 0.15, fy = 0.15)
img4 = cv2.resize(img4, (0, 0), fx = 0.15, fy = 0.15)
width = 4*img1.shape[1]
hieght = img1.shape[0]

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
keypoints3, descriptors3 = sift.detectAndCompute(img3, None)
keypoints4, descriptors4 = sift.detectAndCompute(img4, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
best_matches12 = bf.match(descriptors1, descriptors2)
best_matches34 = bf.match(descriptors3, descriptors4)
raw_matches12 = sorted(best_matches12, key = lambda x:x.distance)
raw_matches34 = sorted(best_matches34, key = lambda x:x.distance)
# construct the two sets of points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in raw_matches12])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in raw_matches12])
points3 = np.float32([keypoints3[m.queryIdx].pt for m in raw_matches34])
points4 = np.float32([keypoints4[m.trainIdx].pt for m in raw_matches34])
(Homography_matrix12, status) = cv2.findHomography(points2, points1, cv2.RANSAC)
(Homography_matrix34, status) = cv2.findHomography(points4, points3, cv2.RANSAC)
print(Homography_matrix12)
print(Homography_matrix34)
stitched_img_src12 = cv2.warpPerspective(img2, Homography_matrix12, (width, hieght))
stitched_img_src34 = cv2.warpPerspective(img4, Homography_matrix34, (width, hieght))
stitched_img_src12[0:img1.shape[0], 0:img1.shape[1]]=img1
stitched_img_src34[0:img3.shape[0], 0:img3.shape[1]]=img3
cropped_stiched_img12 = black_edge_remove(stitched_img_src12)
cropped_stiched_img34 = black_edge_remove(stitched_img_src34)
# cv2.imshow('stitched_image12', cropped_stiched_img12)
# cv2.imshow('stitched_image34', cropped_stiched_img34)

img12 = copy.deepcopy(cropped_stiched_img12)
img34 = copy.deepcopy(cropped_stiched_img34)
gray_img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
gray_img34 = cv2.cvtColor(img34, cv2.COLOR_BGR2GRAY)
keypoints12, descriptors12 = sift.detectAndCompute(gray_img12, None)
keypoints34, descriptors34 = sift.detectAndCompute(gray_img34, None)
best_matches1234 = bf.match(descriptors12, descriptors34)
raw_matches1234 = sorted(best_matches1234, key = lambda x:x.distance)


# keypoints12=np.float32([keypoint.pt for keypoint in keypoints12])
# keypoints34=np.float32([keypoint.pt for keypoint in keypoints34])
# pick up the good keypoints
good_points = []
for j in range(len(raw_matches1234) - 1):
    if raw_matches1234[j].distance < 0.99*raw_matches1234[j + 1].distance:
        good_points.append(raw_matches1234[j])
img_draw = cv2.drawMatches(img12,keypoints12,img34,keypoints34,good_points,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('img_draw', img_draw)
points12 = np.float32([keypoints12[m.queryIdx].pt for m in good_points])
points34 = np.float32([keypoints34[m.trainIdx].pt for m in good_points])
(Homography_matrix1234, status) = cv2.findHomography(points34, points12, cv2.RANSAC)
stitched_img_src1234 = cv2.warpPerspective(img34, Homography_matrix1234, (width, hieght))
# cv2.imshow('stitched_image1234', stitched_img_src1234)
stitched_img_src1234[:img12.shape[0], :img12.shape[1]]=img12
cropped_stiched_img1234 = black_edge_remove(stitched_img_src1234)
cropped_stiched_img1234 = cv2.resize(cropped_stiched_img1234, (0, 0), fx = 0.8, fy = 1)
cv2.imshow('stitched_image', cropped_stiched_img1234)

cv2.waitKey(0)
cv2.destroyAllWindows() 