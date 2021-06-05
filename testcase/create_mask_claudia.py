import cv2
import numpy as np
import numbers
from warnings import warn
import sys
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

patch_size = 20
sigma =20
window_shape = (patch_size, patch_size)
window_stride = 3
num_dict=8000
ksvd_iter = 5
max_sparsity = 3

s=window_stride
orimg = cv2.imread("claudiaorig.png", 0)
image = cv2.imread("claudiamasked.png", 0)
#orimg = cv2.resize(orimg,(256,256))
#image = cv2.resize(image,(128,128))
r,c=orimg.shape
print(r)
print(c)
maskimg=orimg.copy()

j=0
while j+5<254:
    #loc1=np.linspace(j,j+5,num=6,endpoint=True).astype(np.uint8)
    i=0
    while i+5<256:
        #loc2=np.linspace(i,i+5,num=6,endpoint=True).astype(np.uint8)
        maskimg[j:j+5,i:i+5]=0
        i=(i+6)+6
    j=(j+6)+6

j=6
while j+6<=254:
    #loc1=np.linspace(j,j+5,num=6,endpoint=True).astype(np.uint8)
    i=6
    while i+6<=256:
        #loc2=np.linspace(i,i+5,num=6,endpoint=True).astype(np.uint8)
        maskimg[j:j+5,i:i+5]=0
        i=(i+6)+6
    j=(j+6)+6


plt.imshow(maskimg,"gray")
plt.show()

cv2.imwrite("claudiamasked1.png",maskimg.astype('uint8'))


'''
image=image[0:254,:]
print('real',orimg)
print('mask',image)
r,c = image.shape
mask = np.ones((r,c))*255
mask1 = np.ones((r,c))*255
mask[image!=0] = 255
mask1[abs(image-orimg)>10] = 0
#mask=(orimg-image).astype(np.uint8)
plt.imshow(mask,"gray")
plt.show()

plt.imshow(mask1,"gray")
plt.show()
'''