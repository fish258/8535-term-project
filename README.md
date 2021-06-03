# 8535-term-project
Compress Sensing for image denoising&amp;inpainting

# Intro to code

## 1. denoising.ipynb
Remove noise from noisy pictures

## 2. Inpainting.ipynb
Restore Image's color


## 3. some parameters
1. patch_size - Integer, the size of square sliding window
2. sigma - Sigma for Gaussian Noise
3. window_stride - Integer, the stride of sliding window
4. num_dict - the number of columns of dictionary D.
5. ksvd_iter - Iteration number of Ksvd (E step + M sep)
6. max_sparsity - the max sparsity of sparse vector x.

## configure
### 1. denoising
image: read image in.
noise_layer: generate the guassian noise

### 2. inpainting
orimg: the origin image
image: image with missing part.   (orimg!=image is the mask matrix.)