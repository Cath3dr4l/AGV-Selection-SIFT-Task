import numpy as np
import scipy.io
import glob
from selectRegion import roipoly
import pylab as pl
import imageio.v2 as imageio
from dist2 import dist2
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches

framesdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/frames/'
siftdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

fnames.sort()

kmeans = scipy.io.loadmat('kmeans_centers.mat')['kmeans']
BOW = scipy.io.loadmat('BOWV.mat')['BOW']
BOWV = scipy.io.loadmat('BOWV.mat')['BOWV']
refBOW = np.zeros(kmeans.shape[0])

frame_id = [4234]

for id in frame_id:

  mat = scipy.io.loadmat(siftdir + fnames[id])
  im_name = framesdir + fnames[id][:-4]
  im = imageio.imread(im_name)

  pl.imshow(im)
  MyROI = roipoly(roicolor='r')
  RegInd = MyROI.getIdx(im, mat['positions'])

  distance = dist2(kmeans, mat['descriptors'][RegInd,:])
  min_index = np.argmin(distance, axis=1)

  for m in range(len(kmeans)):
    refBOW[m] = np.count_nonzero(min_index == m)

  refBOW = refBOW / np.linalg.norm(refBOW)

Ind = RegInd

fig=plt.figure()
ax=fig.add_subplot(111)
ax.imshow(im)
coners = displaySIFTPatches(mat['positions'][Ind,:], mat['scales'][Ind,:], mat['orients'][Ind,:])

for j in range(len(coners)):
  ax.plot([coners[j][0][0][1], coners[j][1][0][1]], [coners[j][0][0][0], coners[j][1][0][0]], color='g', linestyle='-', linewidth=1)
  ax.plot([coners[j][1][0][1], coners[j][2][0][1]], [coners[j][1][0][0], coners[j][2][0][0]], color='g', linestyle='-', linewidth=1)
  ax.plot([coners[j][2][0][1], coners[j][3][0][1]], [coners[j][2][0][0], coners[j][3][0][0]], color='g', linestyle='-', linewidth=1)
  ax.plot([coners[j][3][0][1], coners[j][0][0][1]], [coners[j][3][0][0], coners[j][0][0][0]], color='g', linestyle='-', linewidth=1)

ax.set_xlim(0, im.shape[1])
ax.set_ylim(0, im.shape[0])  
plt.gca().invert_yaxis()
plt.show()   

refBOW = np.array([refBOW])
print(refBOW)
print(BOWV)

distance = dist2(refBOW,BOWV)
print(distance)
sortedInd = np.argsort(distance[0])
print(sortedInd)

testFrame = np.arange(1,9999)
matches = [testFrame[i] for i in sortedInd[:5]]
print(matches)

plt.figure()
plt.subplot(2,3,1)
plt.imshow(imageio.imread(framesdir + fnames[frame_id[0]][:-4]))
plt.title(f"Original Image {frame_id[0]}")

for j in range(5):
  plt.subplot(2,3,j+2)
  plt.imshow(imageio.imread(framesdir + fnames[matches[j]][:-4]))
  plt.title(f"Similar Image {matches[j]}")

plt.show()