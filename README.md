# 8535-term-project
Compress Sensing for image denoising&amp;inpainting

# Intro to code
## 1. some parameters
1. patch_size - Integer, the size of square sliding window
2. sigma - Sigma for Gaussian Noise
3. window_stride - Integer, the stride of sliding window
4. num_dict - the number of columns of dictionary D.
5. ksvd_iter - Iteration number of Ksvd (E step + M sep)
6. max_sparsity - the max sparsity of sparse vector x.

## jupyter ipynb files
1. denoising.ipynb
Functions of removing noise from noisy pictures
2. Inpainting.ipynb
Functions of Restore Image's color
3. Inpainting DCT.ipynb
Change initiation of Dictionary with DCT.

## python test case files
1.inpainting_gray_lena.py: 
test of example 3

2.inpainting_gray_claudia.py: 
test of example 2

3.inpainting_gray_girls.py:
test of example 4


this case is a given mask case. So the step of how to
get mask is slightly different from the inpainting above


4.inpainting_colored_parrot.py:
test of example 5


in order to test this case. we need to first run this py for 
each color channel. And then use 'merge_three_channel.py' to
get a color inpainting image.


5.inpainting_colored_glasses.py:
test of question 4.3

similar to 4

## configure
### 1. denoising
image: read image in.
noise_layer: generate the guassian noise

### 2. inpainting
orimg: the origin image
image: image with missing part.   (orimg!=image is the mask matrix.)
