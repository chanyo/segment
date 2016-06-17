################################################
## Young Hwan Chang
## chanyo@ohsu.edu
## Do not distribute this CODE
################################################


from __future__ import division
from optparse import OptionParser
import numpy as np 
from skimage import io
from matplotlib import image
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import time
from math import pi
from skimage.segmentation import mark_boundaries
from skimage import exposure #, img_as_float
import cv2


image = '/Users/chanyo/Documents/OHSU_Research/Matched_IF_python/Tiled_HE/HE_2_3.png'
        
start_time = time.time()


parser = OptionParser()
parser.add_option("-i", "--input_image", dest="image", type="string", action="store")
parser.add_option("-o", "--output_directory", dest="output_dir", type="string", action="store")
parser.add_option("-e", "--extension", dest="ext", type="string", action="store")
parser.add_option("-g", "--outlines", dest="outlines", type="string", action="store")




I = io.imread(image).astype(np.uint8) 
R = I[:,:,0]
Ilab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
Ich = cv2.cvtColor(Ilab, cv2.COLOR_RGB2GRAY)
#Ihq = exposure.equalize_adapthist(Igray, nbins=256)
#Ich = cv2.GaussianBlur(Igray,(5,5),0)



deg_rads = np.arange(0.0,180.0,45.0)*pi/180.0
ksize = 8
sigma = 4
lambd = 2
psi = 0

Gfilters = []
for theta in deg_rads:
    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0,  psi, cv2.CV_64F) 
    kern /= 1.5*kern.sum()
    Gfilters.append(kern)

## Gabor Response
gabormag = []
for kern in Gfilters:
    accum = np.zeros_like(Ich)
    fimg = cv2.filter2D(Ich, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    gabormag.append(accum)
    
## Add intensity channel
gabormag.append(Ich)


## Define feature
X = np.asarray(gabormag)
[n_feat, n_x, n_y] = X.shape
X_flat = np.reshape(X, (n_feat, n_x*n_y))
Feature = scale(X_flat, axis=1) # Normalization


# Kmean 
clusters = 4  # Num of cluster
mkb = KMeans(clusters)
mkb.fit(np.transpose(Feature))
L = mkb.labels_.reshape(n_x, n_y)


# Label     
meanI = []
for idx in range(clusters):
    mask_k = R.copy()
    mask_k[L != idx] = 0
    meanI.append( np.mean(mask_k[np.nonzero(mask_k)]))
    
id_sort = sorted(range(clusters),key=lambda x:meanI[x], reverse=False)
mask = np.zeros_like(R)
mask[L == id_sort[0]] = 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
mask_final = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


I_bd = mark_boundaries(I, mask_final, color=(0, 1,1))


io.imsave('/Users/chanyo/Documents/OHSU_Research/Matched_IF_python/mask.png', mask_final, cmap="gray")   
io.imsave('/Users/chanyo/Documents/OHSU_Research/Matched_IF_python/overlay.png', I_bd)   
   


