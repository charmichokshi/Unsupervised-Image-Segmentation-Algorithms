import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

img = cv2.imread('F:/Charmi/19-6 img seg algo codes/city2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

Z = np.float32(img.reshape((-1,3)))
db = DBSCAN(eps=1.2, min_samples=70).fit(Z[:,:2])
# db = DBSCAN(eps=0.3, min_samples=100, metric = 'euclidean',algorithm ='auto')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 10))

ax[0].imshow(img)
ax[1].imshow(np.uint8(db.labels_.reshape(img.shape[:2])))
for a in ax:
    a.axis('off')

fp="F:/Charmi/19-6 img seg algo codes/DBSCAN/"
fn="op.jpg"
plt.savefig(fp+fn) 
plt.show()
