#!/home/sal/Codes/gits/Cartoonify-Image/cartoon-env/bin/python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv.imread('therock.png')
print(type(img))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

line_size = 7
blur_value = 7

gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gray_blur = cv.medianBlur(gray_img, blur_value)
edges = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, line_size, blur_value)

k = 7
data = img.reshape(-1, 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
img_reduced = kmeans.cluster_centers_[kmeans.labels_]
img_reduced = img_reduced.reshape(img.shape)
img_reduced = img_reduced.astype(np.uint8)

blurred = cv.bilateralFilter(img_reduced, d=7, sigmaColor=200,sigmaSpace=200)
cartoon = cv.bitwise_and(blurred, blurred, mask=edges)

cartoon_ = cv.cvtColor(cartoon, cv.COLOR_RGB2BGR)
cv.imwrite('cartoon.png', cartoon_)
cv.imshow('cartoon', cartoon_)
cv.waitKey(0)