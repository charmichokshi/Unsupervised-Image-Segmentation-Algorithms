import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

i = cv2.imread('F:/Charmi/input.tif')
ii = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
plt.imshow(ii)
plt.show()

n = 0
while(n<2):
    img = cv2.pyrDown(img)
    n = n+1
print(1)

Z = np.float32(img.reshape((-1,3)))
img1 = cv2.pyrMeanShiftFiltering(img, 5, 10, 2)
img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2RGB)

cv2.imwrite("F:/Charmi/MEAN SHIFT/op.tif",img1)
