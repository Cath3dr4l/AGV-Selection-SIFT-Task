import numpy as np
import scipy.io
import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
import pylab as pl
import pdb
import imageio.v2 as imageio
import cv2
from sklearn.cluster import KMeans
from dist2 import dist2
import joblib


framesdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/frames/'
siftdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

fnames.sort()

desc_mat = []
position_mat = []
scale_mat = []
orient_mat = []
im_id = []

for i,fname in enumerate(fnames):
  mat = scipy.io.loadmat(siftdir + fname)
  print(mat.keys())
  im_name = framesdir + fname[:-4]
  print(im_name)
  #im = imageio.read(im_name)
  im = cv2.imread(im_name)
  
  if len(mat['descriptors']) < 30:
    sample_desc = len(mat['descriptors'])
  else:
    sample_desc = 30

  ind = np.random.permutation(sample_desc)

  desc_mat.append(mat['descriptors'][ind,:])
  position_mat.append(mat['positions'][ind,:])
  scale_mat.append(mat['scales'][ind,:])
  orient_mat.append(mat['orients'][ind,:])
  im_id.extend([i]*sample_desc)


desc_mat = np.vstack(desc_mat, dtype=np.float32)
print(desc_mat.shape)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000000, 1.0)
# compactness,memberships,kmeans_centers = cv2.kmeans(desc_mat,1500,None,criteria,100,flags=cv2.KMEANS_RANDOM_CENTERS)

# kmeans = KMeans(n_clusters=1500,random_state=42)
# kmeans.fit(desc_mat)

# joblib.dump(kmeans, 'kmeans.joblib')

kmeans = joblib.load('kmeans.joblib')

kmeans_centers = kmeans.cluster_centers_
memberships = kmeans.labels_

words = [400,668,1268]

for word in words:

  numberOfPatches = 0
  matchedIndices = []
  fig = plt.figure()

  for fname in fnames:

    print(fname)

    data = scipy.io.loadmat(siftdir + fname)

    desc = data['descriptors']
    pos = data['positions']
    scale = data['scales']
    orient = data['orients']
    imName = framesdir + fname[:-4]

    distance = dist2(kmeans_centers, desc)

    if(distance.size==0):
      continue

    min_index = np.argmin(distance, axis=1)
    Ind = np.where(min_index == word)[0]

    print(Ind)

    for k in Ind:

      if(k>=len(pos)):
        continue

      im = imageio.imread(imName)

      patch = getPatchFromSIFTParameters(pos[k], scale[k], orient[k], rgb2gray(im))
      
      numberOfPatches += 1
      plt.subplot(5,5,numberOfPatches)
      plt.imshow(patch,cmap='gray')
      
      if numberOfPatches == 25:
        break
    
    print(numberOfPatches)

    if numberOfPatches == 25:
      break


plt.show()

scipy.io.savemat('kmeans_centers.mat', {'kmeans':kmeans_centers})
scipy.io.savemat('kmeans.mat', {'kmeans':kmeans_centers, 'memberships':memberships, 'position_mat':position_mat, 'scale_mat':scale_mat, 'orient_mat':orient_mat, 'im_id':im_id})




    