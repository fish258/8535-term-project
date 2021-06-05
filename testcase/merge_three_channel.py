import cv2
import numpy as np

b=cv2.imread('a2inpainting_b.png',0)#('parrot_originalinpainting_b.png',0)
g=cv2.imread('a2inpainting_g.png',0)#('parrot_originalinpainting_g.png',0)
r=cv2.imread('a2inpainting_r.png',0)#('parrot_originalinpainting_r.png',0)

image=cv2.merge((b,g,r)).astype(np.uint8)

image1=cv2.imread('a2.png')#('parrotinpainted.png')
image1=cv2.resize(image1,(128,128))
cv2.imwrite('glasses_origin_01.png',image1)#('parrot_good_01.png')
cv2.imwrite('glasses_inpainting_01.png',image)#('parrot_inpainting_01.png')
cv2.imshow('i1',image)
cv2.waitKey(0)
cv2.destroyAllWindows()