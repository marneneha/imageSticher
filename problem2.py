import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import math
# from RemoveBlackEdge import remove_black_edge
def black_edge_remove(img):
    for i in range(img.shape[1]):
        if(np.sum(img[:, i])==0):
            break
    cropped_img = img[:, 0:i]
    return cropped_img

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

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# keypoints4, descriptors4 = sift.detectAndCompute(img_gray4, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
best_matches12 = bf.match(descriptors1, descriptors2)
raw_matches12 = sorted(best_matches12, key = lambda x:x.distance)
#change here
keypoints1=np.float32([keypoint.pt for keypoint in keypoints1])
keypoints2=np.float32([keypoint.pt for keypoint in keypoints2])
# construct the two sets of points
points1 = np.float32([keypoints1[m.queryIdx] for m in raw_matches12])
points2 = np.float32([keypoints2[m.trainIdx] for m in raw_matches12])
(Homography_matrix12, status) = cv2.findHomography(points2, points1, cv2.RANSAC)
print(Homography_matrix12)
stitched_img_src = cv2.warpPerspective(img2, Homography_matrix12, (width, hieght))
# stitched_img[0:img1.shape[0], 0:img1.shape[1]]=img1
M1 = np.float32([[1, 0, 0], [0, 1, 0]])
dst1 = cv2.warpAffine(img1, M1, (width, hieght))
dst = cv2.add(dst1, stitched_img_src)
dst_target = np.maximum(dst1, stitched_img_src)
cropped_stiched_img = black_edge_remove(dst_target)
cv2.imshow('stitched_image', cropped_stiched_img)

keypoints3, descriptors3 = sift.detectAndCompute(img3, None)
keypoints12, descriptors12 = sift.detectAndCompute(cropped_stiched_img, None)
best_matches123 = bf.match(descriptors3, descriptors12)
raw_matches123 = sorted(best_matches123, key = lambda x:x.distance)
keypoints12=np.float32([keypoint.pt for keypoint in keypoints12])
keypoints3=np.float32([keypoint.pt for keypoint in keypoints3])

# construct the two sets of points
points3 = np.float32([keypoints3[m.queryIdx] for m in raw_matches123])
points12 = np.float32([keypoints12[m.trainIdx] for m in raw_matches123])
(Homography_matrix123, status) = cv2.findHomography(points12, points3, cv2.RANSAC)
stitched_img_src123 = cv2.warpPerspective(cropped_stiched_img, Homography_matrix123, (width, hieght))
dst3 = cv2.warpAffine(img3, M1, (width, hieght))
dst123 = cv2.add(dst3, stitched_img_src123)
dst_target123 = np.maximum(dst3, stitched_img_src123)
cropped_stiched_img1 = black_edge_remove(dst_target123)
cv2.imshow('stitched_image1', cropped_stiched_img1)

cv2.waitKey(0)
cv2.destroyAllWindows() 