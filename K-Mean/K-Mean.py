import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

img = cv2.imread('1.tif')
fp="F:/Charmi/K-MEAN/"

# ip = cv2.cvtColor(ip, cv2.COLOR_BGR2LAB)
# plt.imshow(img)
# plt.show()

# uncomment to do down-sampling
# n = 0
# while(n<1):
#     img = cv2.pyrDown(img)
#     n = n+1
# rows, cols, chs = img.shape

# ip=np.reshape(img, [-1, 3])
Z = np.float32(img.reshape((-1,3)))

# Define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(Z, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make the original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = np.uint8(label.reshape(img.shape[:2]))
# print(res2.shape)
res2=cv2.resize(res2,(4148,, 3110))
print(res2.shape)
# res2 = cv2.cvtColor(res2, cv2.COLOR_RGB2RGB)

cv2.imwrite("F:/Charmi/K-MEAN/op.tif",res2)

plt.imshow(res2)
plt.show()
