
# coding: utf-8

# In[1]:

'''
Ksvd method - Paper "Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries " 
(available at https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnuber=4011956&tag=1)
Please note that this implementation follows the workflow of  Deepayan137's implementation (https://github.com/Deepayan137/K-svublob/master/main_vamsi_2.py).

'''
import cv2
import numpy as np
import numbers
from warnings import warn
import sys
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

# some parameters
patch_size = 7
sigma =20
window_shape = (patch_size, patch_size)
window_stride = 5
num_dict=1000
ksvd_iter = 2
max_sparsity = 10


# # Some Functions

# In[2]:

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.   If `arr_in` is
        non-contiguous, a copy is made.
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    if not arr_in.flags.contiguous:
        warn(RuntimeWarning("Cannot provide views on a non-contiguous input "
                            "array without copying."))

    arr_in = np.ascontiguousarray(arr_in)

    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def getImagePatches(img, stride):
    '''
    Input：
        - img: image matrix
        - stride：滑动步幅
    Return：
        - patches - (m,n) √m*√m-size's sliding window sampling patches.
            - m: window size vec
            - n: number of samples
        - patch - (r, c, √m, √m)
            - r: number of window patches on vertical direction
            - c: number of window patches on horizontal direction
            - √m: window size
            e.g. - (44,44, 7,7) 有44*44个patches，每个patches都是7*7的
    '''
    # get indices of each patch from image matrix
    # This is also the R_{ij} marix mentioned in the paper
    patch_indices = view_as_windows(img, window_shape, step=stride) # 返回 (r,c)个(window_shape)的数组(94,94,7,7)

    r,c= patch_indices.shape[0:2]  # window matrix size: 有r*c个滑动窗 （44，44）个
    i_r,i_c=patch_indices.shape[2:4]  # image patch size: 窗口的大小（7，7）的
    patches = np.zeros((i_r*i_c, r*c)) # (49,8836)，每一列都是一个滑动窗
    # extract each image patchlena.png
    for i in range(r):
        for j in range(c):
            # 拉平每个patch到vec -〉 (7*7, 44*44)
            patches[:, j+patch_indices.shape[1]*i] = np.concatenate(patch_indices[i, j], axis=0)
    return patches, patch_indices.shape

def reconstruct_image(patch_final, noisy_image):
    '''
    :param patch_final: recovered patches i.g. (m, num)
    :param noisy_image: noisy image
    :return:
        img_out: denoised image
    '''
    img_out = np.zeros(noisy_image.shape)
    weight = np.zeros(noisy_image.shape)
    num_blocks = noisy_image.shape[0] - patch_size + 1
    for l in range(patch_final.shape[1]):
        i, j = divmod(l, num_blocks)
        temp_patch = patch_final[:, l].reshape(window_shape)
        # img_out[i, j] = temp_patch[1, 1]
        img_out[i:(i+patch_size), j:(j+patch_size)] = img_out[i:(i+patch_size), j:(j+patch_size)] + temp_patch
        weight[i:(i+patch_size), j:(j+patch_size)] = weight[i:(i+patch_size), j:(j+patch_size)] + np.ones(window_shape)

    img_out = (noisy_image+0.034*sigma*img_out)/(1+0.034*sigma*weight)

    return img_out


def omp(D, data, sparsity):
    '''
    Given D，Y; Calculate X
    Input:
        D - dictionary (m, n-dims)
        data - Y: all patches (m, num-sample)
        sparsity - sparsity of x
    Output:
        X - sparse vec X: (n-dims, num-sample)
    '''
    X = np.zeros((D.shape[1], data.shape[1]))  # (n-dims, num-sample)
    tot_res = 0   # collect normed residual from every y-D@x
    # go through all data patches
    for i in range(data.shape[1]):
        # for all num-samples, every sample will have k-sparsity
        # every loop, finish one sample x
        ################### process bar ########################
        count = np.floor((i + 1) / float(data.shape[1]) * 100)
        sys.stdout.write("\r- omp Sparse coding : Channel : %d%%" % count)
        sys.stdout.flush()
        #######################################################
        #
        y = data[:, i]  # ith sample y, corresponding to ith x - (m,1)
        res = y  # initial residual
        omega = []
        res_norm = np.linalg.norm(res)
        xtemp_sparse = np.zeros(D.shape[1])  # (500,)

        while len(omega) < sparsity:
            # loop until x has sparsity-sparse (k-sparsity)
            # every loop, find one more sparse element
            proj = D.T @ res  # projection: find the max correlation between residual&D
            i_til = np.argmax(np.abs(proj))  # max correlation column
            omega.append(i_til)
            xtemp_sparse = np.linalg.pinv(D[:,omega])@y   # x = D^-1 @ y
            d_omg = D[:, omega]                  # (m, columns now have)
            recover_alr_y = d_omg @ xtemp_sparse  # y_til now can recover
            res = y - recover_alr_y           # calculate residual left
            res_norm = np.linalg.norm(res)  # update norm residual of this x

        tot_res += res_norm
        # update xi
        if len(omega) > 0:
            X[omega, i] = xtemp_sparse
    print('\r Sparse coding finished.\n')
    return X


def dict_initiate(patches, dict_size):
    '''
    dictionary intialization
    assign data columns to dictionary at random
    :param patches: (m, num of samples)
    :param dict_size: n-dims - then this would be the dimension of sparse vector x
    :return:
    D: normalized dictionary D
    '''
    # random select n-dims columns index
    indices = np.random.random_integers(0, patches.shape[1] - 1, dict_size)  # (500,)
    # choose the n-dims columns in Y as initial D
    D = np.array(patches[:, indices])  # select n-dims patches

    return D - D.mean()  # return normalized dictionary


# update dictionary and sparse representations after sparse coding
def dict_update(D, data, X, j):
    '''
    Input:
        D - Dictionary (m, n-dims)
        data - Y all patches (m, num of samples)
        X: sparse matrix for x。(n-dims, num of samples) 每个patch变成了500维的稀疏向量，有8836个patch。
        j: now update the jth column of D
    Output:
        D_temp: new dictionary
        X: X would be updated followed by D
    '''
    indices = np.where(X[j, :] != 0)[0]  # find all x contributed to the i_til column
    D_temp = D  # work on new dictionary
    X_temp = X[:, indices]  # all x contributed to the i_til column

    if len(indices) > 1:
        # there're x contribute to this column
        X_temp[j, :] = 0  # set X's i_til row element to 0. remove the contribute to this column
        # ek: Y - D@X_temp: the contribution only of j column
        e_k = data[:, indices] - D_temp @ X_temp  # (m, a couple of columns)
        # make ek to be 2D matrix. (if only have 1 column, e_k would be a 1d array)
        u, s, vt = np.linalg.svd(np.atleast_2d(e_k))  # u就计算一列，s就一个最大值，vt就一行
        u = u[:,0]
        s = s[0]
        vt = vt[0,:]
        D_temp[:, j] = u  # update dictionary with first column
        X[j, indices] = s * vt  # update x the sparse representations
    else:
        # no x have non-zero element corresponding to this column
        pass

    return D_temp, X


def k_svd(patches, dict_size, sparsity):
    '''
    :param patches: patches from image (m, num of samples)
    :param dict_size: n-dims of every x
    :param sparsity: sparsity of every x
    :return:
        D: final dictionary
        X: corresponding X matrix (perhaps not sparse, so need omp to update again)
    '''
    # initial dictionary D
    D = dict_initiate(patches, dict_size)
    # initializing sparse matrix: X
    X = np.zeros((D.T.dot(patches)).shape)  # (n-dims, num of samples)

    for k in range(ksvd_iter):  # ksvd_iter = 1
        print("KSVD Iter: {}/{} ".format(k + 1, ksvd_iter))
        # E step， update X
        X = omp(D, patches, sparsity)  # (n-dims, num of samples)
        # M step，update D
        count = 1
        dict_elem_order = np.random.permutation(D.shape[1])  # (0 ~ n-dims-1) array
        # get order of column elements
        for j in dict_elem_order:
            # update D column by column
            ################## process bar ###############################
            r = np.floor(count / float(D.shape[1]) * 100)
            sys.stdout.write("\r- k_svd Dictionary updating : %d%%" % r)
            sys.stdout.flush()
            ##############################################################
            # calculate the jth column
            D, X = dict_update(D, patches, X, j)
            count += 1
        print("\nDictionary updating  finished")
    return D, X


def denoising(img_noisy, dict_size, sparsity):
    '''
    Input:
        img_noisy: input image
        dict_size: n-dims
        sparsity: sparsity of x
    Return:
        denoised_image: denoised image
    '''
    # generate noisy patches.

    stride = 1
    # get patches
    patches, patches_shape = getImagePatches(img_noisy, stride)
    mean = patches.mean()
    patches = patches - mean

    # K-SVD.
    dict_final, sparse_init = k_svd(patches, dict_size, sparsity)

    # omp
    noisy_patches, noisy_patches_shape = getImagePatches(img_noisy, stride=1)
    data_mean = noisy_patches.mean()
    noisy_patches = noisy_patches - data_mean

    sparse_final = omp(dict_final, noisy_patches, sparsity)

    # Reconstruct the image.
    patches_approx = np.dot(dict_final, sparse_final) + data_mean
    denoised_image = reconstruct_image(patches_approx, img_noisy)

    return denoised_image


# In[4]:
'''
def main():
    # read iamge (grayscale )
    image = cv2.imread("lena.png", 0)
    image = cv2.resize(image,(128,128))
    # impose noisy to image 
    noise_layer = np.random.normal(0, sigma ^ 2, image.size).reshape(image.shape).astype(int)
    noisy_image = image + noise_layer
    
    
    print('num_dict(设置稀疏向量的长度):',num_dict,'max_sparsity(稀疏向量最大稀疏数):',max_sparsity)
    # denose the given image
    denoised_image = denoising(noisy_image, dict_size=num_dict, sparsity=max_sparsity)

    noisy_psnr = 20*np.log10(np.amax(image)) - 10*np.log10(pow(np.linalg.norm(image - noisy_image), 2)/noisy_image.size)
    final_psnr = 20*np.log10(np.amax(image)) - 10*np.log10(pow(np.linalg.norm(image - denoised_image), 2)/denoised_image.size)
    # save images  
    cv2.imwrite("lena.png".split(".")[0]+str(sigma)+"noisy.jpg", noisy_image.astype('uint8'))
    cv2.imwrite("lena.png".split(".")[0]+str(sigma)+"denoised.jpg", denoised_image.astype('uint8'))
    cv2.imwrite("lena.png".split(".")[0]+str(sigma)+"difference.jpg", np.abs(noisy_image - denoised_image).astype('uint8'))

    print("PSRN: {}(noisy), {}(denoised))".format(noisy_psnr, final_psnr))
    return 0

if __name__=='__main__':
    main()


# In[72]:

plt.rcParams['figure.figsize'] = [5, 5]
i1 = cv2.imread("lena20difference.jpg", 0)
plt.imshow(i1,"gray")
plt.show()


# In[73]:

i2 = cv2.imread("lena20noisy.jpg", 0)
plt.imshow(i2,"gray")
plt.show()


# In[74]:

i3 = cv2.imread("lena20denoised.jpg", 0)
plt.imshow(i3,"gray")
plt.show()


# In[75]:

image = cv2.imread("lena.png", 0)
# image = cv2.resize(image,(100,100))
# impose noisy to image 
noise_layer = np.random.normal(0, sigma ^ 2, 512*512).reshape(image.shape).astype(int)
noisy_image = image + noise_layer

plt.imshow(noisy_image,"gray")
plt.show()


# In[76]:

plt.imshow(image,"gray")
plt.show()


# In[77]:

plt.rcParams['figure.figsize'] = [20, 10]
plt.subplot(1,3,1)
plt.title("origin")
plt.imshow(cv2.resize(image,(128,128)),"gray")
plt.subplot(1,3,2)
plt.title("noisy")
plt.imshow(i2,"gray")
plt.subplot(1,3,3)
plt.title("denoised")
plt.imshow(i3,"gray")
plt.show()


# In[ ]:
'''


