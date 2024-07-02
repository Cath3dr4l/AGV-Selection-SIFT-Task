import cv2
import numpy as np
import scipy
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
import pylab as pl
from dist2 import dist2

framesdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/frames/'
siftdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/sift/'
wdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/'

fname = wdir + 'twoFrameData.mat'

mat = scipy.io.loadmat(fname)
im1_arr = np.array(mat['im1'])
im2_arr = np.array(mat['im2'])

im1 = np.array(im1_arr, dtype=np.uint8)
im2 = np.array(im2_arr, dtype=np.uint8)

pl.imshow(im1)
MyROI = roipoly(roicolor='r')
Ind = MyROI.getIdx(im1, mat['positions1'])

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.imshow(im1)
# coners = displaySIFTPatches(mat['positions1'][Ind,:], mat['scales1'][Ind,:], mat['orients1'][Ind,:])

# for j in range(len(coners)):
#   ax.plot([coners[j][0][0][1], coners[j][1][0][1]], [coners[j][0][0][0], coners[j][1][0][0]], color='g', linestyle='-', linewidth=1)
#   ax.plot([coners[j][1][0][1], coners[j][2][0][1]], [coners[j][1][0][0], coners[j][2][0][0]], color='g', linestyle='-', linewidth=1)
#   ax.plot([coners[j][2][0][1], coners[j][3][0][1]], [coners[j][2][0][0], coners[j][3][0][0]], color='g', linestyle='-', linewidth=1)
#   ax.plot([coners[j][3][0][1], coners[j][0][0][1]], [coners[j][3][0][0], coners[j][0][0][0]], color='g', linestyle='-', linewidth=1)

# ax.set_xlim(0, im1.shape[1])
# ax.set_ylim(0, im1.shape[0])  
# plt.gca().invert_yaxis()
# plt.show()    

selectedDescriptors1 = mat['descriptors1'][Ind,:]
descriptors2 = mat['descriptors2']

print(selectedDescriptors1)
print(descriptors2)

matches = dist2(descriptors2, selectedDescriptors1)

print(matches.shape)

matches = matches.min(axis=1)
print(matches.shape)


Ind2 = np.where(matches<0.15)[0]

# print(type(Ind),Ind)
# print(type(Ind2),Ind2)

fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(im2)
coners = displaySIFTPatches(mat['positions2'][Ind2,:], mat['scales2'][Ind2,:], mat['orients2'][Ind2,:])

for j in range(len(coners)):
  bx.plot([coners[j][0][0][1], coners[j][1][0][1]], [coners[j][0][0][0], coners[j][1][0][0]], color='g', linestyle='-', linewidth=1)
  bx.plot([coners[j][1][0][1], coners[j][2][0][1]], [coners[j][1][0][0], coners[j][2][0][0]], color='g', linestyle='-', linewidth=1)
  bx.plot([coners[j][2][0][1], coners[j][3][0][1]], [coners[j][2][0][0], coners[j][3][0][0]], color='g', linestyle='-', linewidth=1)
  bx.plot([coners[j][3][0][1], coners[j][0][0][1]], [coners[j][3][0][0], coners[j][0][0][0]], color='g', linestyle='-', linewidth=1)

bx.set_xlim(0, im2.shape[1])
bx.set_ylim(0, im2.shape[0])  
plt.gca().invert_yaxis()
plt.show()    
