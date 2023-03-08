import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import math
import glob

image_paths = glob.glob('problem_2_images/*.jpg')
images = []
sift = cv2.SIFT_create()
for image in range(0, len(image_paths)-1):
    img1 = cv2.imread(image_paths[image])
    img2 = cv2.imread(image_paths[image+1])
    img1 = cv2.resize(img1, (0, 0), fx = 0.2, fy = 0.2)
    img2 = cv2.resize(img2, (0, 0), fx = 0.2, fy = 0.2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    keypoints1, descriptors1 = sift.detectAndCompute(img_gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_gray2, None)
    img_gray1 = cv2.drawKeypoints(img_gray1, keypoints1,None)
    img_gray2 = cv2.drawKeypoints(img_gray2, keypoints2,None)
    cv2.imshow('gray image1', img_gray1)
    cv2.imshow('gray image2', img_gray2)
    #done until here
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_matches = bf.match(descriptors1, descriptors2)
    raw_matches = sorted(best_matches, key = lambda x:x.distance)
    #change here
    keypoints1=np.float32([keypoint.pt for keypoint in keypoints1])
    keypoints2=np.float32([keypoint.pt for keypoint in keypoints2])
    if len(raw_matches) > 4:
    # construct the two sets of points
        points1 = np.float32([keypoints1[m.queryIdx] for m in raw_matches])
        points2 = np.float32([keypoints2[m.trainIdx] for m in raw_matches])
        (Homography_matrix, status) = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(Homography_matrix)
    width = 2*img1.shape[1]
    hieght = img1.shape[0]
    stitched_img = cv2.warpPerspective(img1, Homography_matrix, (width, hieght))
    stitched_img[0:img2.shape[0], 0:img2.shape[1]]=img2
    cv2.imshow('stitched_image', stitched_img)
    cv2.waitKey(0)
# imageStritcher = cv2.Stitcher_create()
# error, stitched_img = imageStritcher.stitch(images)
# if not error:
#     cv2.imwrite("stitcherOutput.png", stitched_img)
#     cv2.imshow("stiched Image", stitched_img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows() 