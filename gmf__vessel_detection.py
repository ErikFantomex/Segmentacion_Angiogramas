# -*- coding: utf-8 -*-
"""GMF__vessel_detection.py
"""

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from skimage.transform import rotate
from skimage.io import imread
from math import sin,cos,acos,pi,sqrt,asin
from skimage.transform import radon, rescale
import numpy as np
import math
import cv2 
from skimage.filters import try_all_threshold
from matplotlib import pyplot as plt 
import numpy.matlib


# El chilo:
def GMF(Y, sigma, L, T, K):
    M,N = np.shape(Y)
    x = np.arange(-math.floor(T/2),math.floor(T/2)+1)
    tmp1 = np.exp(-(x*x)/(2*sigma*sigma))
    tmp1 = max(tmp1)-tmp1
    ht1 = np.matlib.repmat(tmp1,L,1)
    sht1 = np.sum(ht1)
    mean = sht1/(T*L)
    ht1 = ht1 - mean
    ht1 = ht1/sht1

    h = []
    h.append(np.zeros((L+6, T+3)))
    h[0][3:L+3,1:T+1] = ht1
    for k in range(1,K):
        ag = (180/K)*k
        h.append(rotate(h[0],angle=ag,order=3, resize=False))

    R = [convolve2d(Y, hi, mode='same') for hi in h]

    rt = np.zeros((M,N))
    ER = np.zeros(K)

    for i in range(M):
        for j in range(N):
            for f in range(K):
                ER[f] = R[f][i,j]
            rt[i,j] = max(ER)

    rmin = np.abs(np.min(rt))
    rt = rt + rmin

    rmax = np.max(rt)
    rt = np.round(rt*255.0/rmax)

    return rt

img = 'Database_134_Angiograms/33.pgm'

Y = cv2.imread(img,0)
sigma = 1.8
L= 16
T= 17
K = 12

rt = GMF(Y, sigma, L, T, K)
plt.imshow(rt)


#Agrupamos todas imagenes en una lista 
#import os 

#images=[]

#def getFiles(path):
#    for file in os.listdir(path):
#        if file.endswith(".pgm"):
#            images.append(os.path.join(path, file))

#filesPath = 'Database_134_Angiograms'

#getFiles(filesPath)
#print(images)


# Vamos a definir una nueva funcion para enseñar imagenes

#def imshow(title="Image", image = None, size = 10):
#  w, h = image.shape[0], image.shape[1]
#  aspect_ratio = w/h
#  plt.figure(figsize=(size*aspect_ratio,size))
#  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#  plt.title(title)
#  plt.show()


#contador = 0
#for i in images:
#  contador = contador + 1
#  imagen = cv2.imread(i,0)
  #gass = GMF(imagen,sigma,L,T,K)
  #plt.imshow(gass)
#  imshow(str(contador),imagen)

#imshow(rt)
#rt
#plt.imshow(rt)
#rt.shape

#contador = 0 
#for i in images:
#Y = cv2.imread(img,0)
#sigma = 1.8
#L= 16
#T= 17
#K = 12
#rt = GMF(Y, sigma, L, T, K)
#  contador = contador + 1
#plt.imshow(GMF(Y, sigma, L, T, K))

#from skimage import data

#from skimage.filters import threshold_otsu
fig, ax = try_all_threshold(rt, figsize=(10, 8), verbose=False)
plt.show()

image = rt
thresh = threshold_otsu(image)
binary = image > thresh

#skeletonizacion 
skeleton = skeletonize(binary)
# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(binary, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()

plt.imshow(skeleton)
skeleton.astype
skeleton


"""# Ramer douglas peucker """
img = cv2.imread('Database_134_Angiograms/33.pgm',0)
zero=np.zeros(img.shape)
zero2=np.zeros(img.shape)
eps=400
def perpendi(a):
    if len(a)<=2:

        return a
    elif len(a)==3:
        x=(a[0][0],a[0][1])
        z=(a[len(a)-1][0],a[len(a)-1][1])
        y=(a[1][0],a[1][1])
        x1=(y[0]-x[0],y[1]-x[1])
        mag_u=sqrt(x1[0]**2+x1[1]**2)
        y1=(z[0]-x[0],z[1]-x[1])
        mag_v=sqrt(y1[0]**2+y1[1]**2)
        w1=(x1[0]*y1[0])+(x1[1]*y1[1])
        w1=w1/mag_v
        w2=sqrt((x1[0]**2+x1[1]**2)-w1**2)
        print(w2)
        if w2<eps:

            return [x,z]
        else:

            return[x,y,z]
    else:
        x=(a[0][0],a[0][1])
        z=(a[len(a)-1][0],a[len(a)-1][1])
        maxx=0
        mid=0
        for i in range(1,len(a)-1):
            y=(a[i][0],a[i][1])
            

            x1=(y[0]-x[0],y[1]-x[1])
            mag_u=sqrt(x1[0]**2+x1[1]**2)
            y1=(z[0]-x[0],z[1]-x[1])
            mag_v=sqrt(y1[0]**2+y1[1]**2)

            w1=(x1[0]*y1[0])+(x1[1]*y1[1])
            w1=w1/mag_v

            w2=sqrt((x1[0]**2+x1[1]**2)-w1**2)

            if w2>maxx:
                maxx=w2
                mid=i
        ret1=perpendi(a[:mid+1])
        ret2=perpendi(a[mid:])
        
        return ret1+ret2
        
        
pts = np.array([(150,150),(150,300),(250,20),(300,90),(400,150),(500,70),(520,60),(550,100),(590,80)], np.int32)
pt=[(150,150),(150,300),(250,20),(300,90),(400,150),(500,70),(520,60),(550,100),(590,80)]
cv2.polylines(zero,[pts],True,(255))
lis=perpendi(pt)

print(lis)
cv2.polylines(zero2,[np.array(lis,np.int32)],True,(255),1)
cv2.imshow(zero)
cv2.imshow(zero2)
k = cv2.waitKey(0)
if k == 27: 
    cv2.destroyAllWindows()


