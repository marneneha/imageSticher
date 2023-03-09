import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import math
# import glob
images = []
# image_paths = glob.glob('problem_2_images/image_1.jpg')
sift = cv2.SIFT_create()
img1 = cv2.imread('problem_2_images/image_1.jpg')
img2 = cv2.imread('problem_2_images/image_2.jpg')
img3 = cv2.imread('problem_2_images/image_3.jpg')
# img4 = cv2.imread('problem_2_images/image_4.jpg')
width = 3*img1.shape[1]
hieght = img1.shape[0]
img1 = cv2.resize(img1, (0, 0), fx = 0.2, fy = 0.2)
img2 = cv2.resize(img2, (0, 0), fx = 0.2, fy = 0.2)
img3 = cv2.resize(img3, (0, 0), fx = 0.2, fy = 0.2)
# img4 = cv2.resize(img4, (0, 0), fx = 0.2, fy = 0.2)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img_gray3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
# img_gray4 = cv2.cvtColor(img4, cv2.COLOR_RGB2GRAY)
keypoints1, descriptors1 = sift.detectAndCompute(img_gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img_gray2, None)
keypoints3, descriptors3 = sift.detectAndCompute(img_gray3, None)
# keypoints4, descriptors4 = sift.detectAndCompute(img_gray4, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
best_matches12 = bf.match(descriptors1, descriptors2)
best_matches23 = bf.match(descriptors2, descriptors3)
# best_matches34 = bf.match(descriptors3, descriptors4)
raw_matches12 = sorted(best_matches12, key = lambda x:x.distance)
raw_matches23 = sorted(best_matches23, key = lambda x:x.distance)
# raw_matches34 = sorted(best_matches34, key = lambda x:x.distance)
#change here
keypoints1=np.float32([keypoint.pt for keypoint in keypoints1])
keypoints2=np.float32([keypoint.pt for keypoint in keypoints2])
keypoints3=np.float32([keypoint.pt for keypoint in keypoints3])
# keypoints4=np.float32([keypoint.pt for keypoint in keypoints4])


if len(raw_matches12) > 4:
# construct the two sets of points
    points1 = np.float32([keypoints1[m.queryIdx] for m in raw_matches12])
    points2 = np.float32([keypoints2[m.trainIdx] for m in raw_matches12])
    (Homography_matrix12, status) = cv2.findHomography(points2, points1, cv2.RANSAC)
    print(Homography_matrix12)
stitched_img = cv2.warpPerspective(img2, Homography_matrix12, (width, hieght))
stitched_img[0:img1.shape[0], 0:img1.shape[1]]=img1

keypoints12, descriptors12 = sift.detectAndCompute(stitched_img, None)
best_matches123 = bf.match(descriptors12, descriptors3)
raw_matches123 = sorted(best_matches123, key = lambda x:x.distance)
keypoints12=np.float32([keypoint.pt for keypoint in keypoints12])

cv2.imshow('stitched_image', stitched_img)

if len(raw_matches23) > 4:
# construct the two sets of points
    points12 = np.float32([keypoints12[m.queryIdx] for m in raw_matches23])
    points3 = np.float32([keypoints3[m.trainIdx] for m in raw_matches23])
    (Homography_matrix123, status) = cv2.findHomography(points3, points12, cv2.RANSAC)
stitched_img = cv2.warpPerspective(img3, Homography_matrix123, (width, hieght))
stitched_img[0:img3.shape[0], 0:img3.shape[1]]=img3
# if len(raw_matches34) > 4:
# # construct the two sets of points
#     points1 = np.float32([keypoints3[m.queryIdx] for m in raw_matches34])
#     points2 = np.float32([keypoints4[m.trainIdx] for m in raw_matches34])
#     (Homography_matrix34, status) = cv2.findHomography(points1, points2, cv2.RANSAC)
# # print(Homography_matrix)

#strich image

# stitched_img = cv2.warpPerspective(stitched_img, Homography_matrix34, (width, hieght))
# stitched_img[0:img4.shape[0], 0:img4.shape[1]]=img4
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows() 