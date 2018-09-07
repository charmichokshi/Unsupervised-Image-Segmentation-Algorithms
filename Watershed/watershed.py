from skimage.segmentation import quickshift as qs
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('F:/Charmi/coffee.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

img = qs(img, convert2lab=True)

fp="F:/Charmi/WATERSHED/"
fn="op.jpg"
plt.savefig(fp+fn) 

plt.imshow(img)
plt.show()
