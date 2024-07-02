import scipy.io
import numpy as np
import glob
from dist2 import dist2
import matplotlib.pyplot as plt

framesdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/frames/'
siftdir = '/Users/shashwatsinghranka/Documents/AGV Tasks/VideoSearch/sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

fnames.sort()

kmeans = scipy.io.loadmat('kmeans_centers.mat')['kmeans']

K = kmeans.shape[0]
BOW = np.zeros((len(fnames), K))
binRange = np.arange(K+1)

# for i,fname in enumerate(fnames):

#   mat = scipy.io.loadmat(siftdir + fname)
#   im_name = framesdir + fname[:-4]

#   descriptors = mat['descriptors']
#   noDesc = mat['descriptors'].shape[0]

#   if (len(mat['descriptors']) != 0):

#     distance = dist2(kmeans, descriptors)
#     index = np.argmin(distance, axis=1)

#     binCounts = np.histogram(index, binRange)[0]
#     BOW[i,:] = binCounts

BOW = scipy.io.loadmat('BOWV.mat')['BOW']
BOWV = scipy.io.loadmat('BOWV.mat')['BOWV']

# BOWV = np.zeros(BOW.shape)

# for i in range(BOW.shape[0]):
#   norm = np.linalg.norm(BOW[i, :])
#   if norm != 0:
#     BOWV[i, :] = BOW[i, :] / norm
#   else:
#     BOWV[i, :] = np.zeros_like(BOW[i, :])

# scipy.io.savemat('BOWV.mat', {'BOW':BOW, 'BOWV':BOWV})

for imID in [1679,4690,5961,1092]:

  tempMat = np.zeros(len(fnames))

  for i in range(len(fnames)):
    tempMat[i] = np.dot(BOWV[imID,:], BOWV[i,:])

  dot_mat = np.zeros(5)
  tempMat[imID] = 0

  for i in range(5):
    dot_mat[i] = np.argmax(tempMat)
    tempMat[int(dot_mat[i])] = 0

  plt.figure()
  plt.subplot(2,3,1)
  plt.imshow(plt.imread(framesdir + fnames[imID][:-4]))
  plt.title(f'Original Frame {imID}')

  for i in range(5):
    plt.subplot(2,3,i+2)
    plt.imshow(plt.imread(framesdir + fnames[int(dot_mat[i])][:-4]))
    plt.title(f'Similar Frame {int(dot_mat[i])}')

plt.show()





