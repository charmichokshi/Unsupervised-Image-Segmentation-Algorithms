from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('F:/Charmi/city.png')
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

labels1 = segmentation.slic(img, compactness=30, n_segments=20000)
out1 = color.label2rgb(labels1, img, kind='avg')
g = graph.rag_mean_color(img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)

out2 = color.label2rgb(labels2, img, kind='avg')
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
ax[0].imshow(img)
# ax[1].imshow(out1)
ax[1].imshow(out2)
for a in ax:
    a.axis('off')


fp="F:/Charmi/N CUT (GRAPH CUT)/"
fn="op.jpg"
plt.savefig(fp+fn) 
plt.show()
