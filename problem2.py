import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import math
import glob
# import imutils

image_paths = glob.glob('problem_2_images/*.jpg')
images = []
for image in image_paths:
    img = cv2.imread(image)
    img = cv2.resize(img, (0, 0), fx = 0.2, fy = 0.2)
    images.append(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)

imageStritcher = cv2.Stitcher_create()
error, stitched_img = imageStritcher.stitch(images)
if not error:
    cv2.imwrite("stitcherOutput.png", stitched_img)
    cv2.imshow("stiched Image", stitched_img)
    cv2.waitKey(0)
# cv2.destroyAllWindows() 