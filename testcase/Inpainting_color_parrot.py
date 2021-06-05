
# coding: utf-8

# In[1]:

import cv2
import numpy as np
import numbers
from warnings import warn
import sys
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

# some parameters
patch_size = 10
sigma =20
window_shape = (patch_size, patch_size)
window_stride = 2
num_dict=5000#
ksvd_iter = 3
max_sparsity = 25#25


# In[6]:

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

def omp(D, data, sparsity, mask):
    '''
    Given D，Y; Calculate X
    Input:
        D - dictionary (m, n-dims)
        data - Y: all patches (m, num-sample)
        sparsity - sparsity of x
        mask - M: mask matrix for all patches (m, num-sample)
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
        mi = mask[:,i]==1 # (m,) of true/false
        D_temp = D[mi,:]  # 选取有效值的行
        y = data[:,i]
        y_true = y[mi] # 不要无效观测值
        res = y_true  # initial residual
        thresh = np.linalg.norm(res)*0.01
        omega = []
        res_norm = np.linalg.norm(res)
        xtemp_sparse = np.zeros(D.shape[1])  # (500,)

        while len(omega) < sparsity and res_norm>thresh:
            # loop until x has sparsity-sparse (k-sparsity)
            # every loop, find one more sparse element
            proj = D_temp.T @ res  # projection: find the max correlation between residual&D
            i_til = np.argmax(np.abs(proj))  # max correlation column
            omega.append(i_til)
            xtemp_sparse = np.linalg.pinv(D_temp[:,omega])@y_true   # x = D^-1 @ y
            d_omg = D_temp[:, omega]                  # (m, columns now have)
            recover_alr_y = d_omg @ xtemp_sparse  # y_til now can recover
            res = y_true - recover_alr_y           # calculate residual left
            res_norm = np.linalg.norm(res)  # update norm residual of this x
        #
        # tot_res += res_norm
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
    indices = np.random.random_integers(0, patches.shape[1] - 1, dict_size)  #
    # choose the n-dims columns in Y as initial D
    D = np.array(patches[:, indices])  # select n-dims patches

    return D - D.mean()  # return normalized dictionary

def dict_update(D, data, X, j, mask):
    '''
    Input:
        D - Dictionary (m, n-dims)
        data - Y all patches (m, num of samples)
        X: sparse matrix for x。(n-dims, num of samples) 每个patch变成了500维的稀疏向量，有8836个patch。
        j: now update the jth column of D
        mask - M: mask matrix for all patches (m, num-sample)
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
        # ek: Y - D@X_temp: the contribution only of j column.
        # mask set 不可见的行为0，即这些行没有造成误差
        e_k = mask[:,indices]*(data[:, indices] - D_temp @ X_temp)  # (m, a couple of columns)
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

def k_svd(patches, dict_size, sparsity, mask):
    '''
    :param patches: patches from image (m, num of samples)
    :param dict_size: n-dims of every x
    :param sparsity: sparsity of every x
    mask - M: mask matrix for all patches (m, num-sample)
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
        X = omp(D, patches, sparsity, mask)  # (n-dims, num of samples)
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
            D, X = dict_update(D, patches, X, j, mask)
            count += 1
        print("\nDictionary updating  finished")
    return D, X

def inpainting(img_degrade, dict_size, sparsity, mask):
    '''
    Input:
        img_noisy: input image
        dict_size: n-dims
        sparsity: sparsity of x
        mask - M: mask matrix for all patches (m, num-sample)
    Return:
        denoised_image: denoised image
    '''
    # generate noisy patches.
    s=window_stride
    # get patches
    patches, patches_shape = getImagePatches(img_degrade, s)
    mean = patches.mean()
    patches = patches - mean

    # K-SVD.
    dict_final, sparse_init = k_svd(patches, dict_size, sparsity,mask)

    # omp
    degrade_patches, degrade_patches_shape = getImagePatches(img_degrade, s)
    data_mean = degrade_patches.mean()
    degrade_patches = degrade_patches - data_mean

    sparse_final = omp(dict_final, degrade_patches, sparsity, mask)

    # Reconstruct the image.
    patches_approx = np.dot(dict_final, sparse_final) + data_mean
    inpainting_image = reconstruct_image(patches_approx, img_degrade, s)

    return inpainting_image

def reconstruct_image(patch_final, degraded_image, stride):
    '''
    :param patch_final: recovered patches i.g. (m, num)
    :param degraded_image: image of degraded
    :param stride: stride when select patches
    :return:
        img_out: denoised image
    '''
    m_size,n_size = degraded_image.shape
    r = (m_size-patch_size)//stride+1
    c = (n_size-patch_size)//stride+1
    img_out = np.zeros((m_size,n_size))
    weight = np.zeros((m_size,n_size))
    print(r,c)
    for i in range(r):
        for j in range(c):
            rpos = i*stride
            cpos = j*stride
            img_out[rpos:rpos+patch_size,cpos:cpos+patch_size] += patch_final[:,i*c+j].reshape(patch_size,patch_size)
            weight[rpos:rpos+patch_size,cpos:cpos+patch_size] += 1
    print(weight)
    img_out = img_out/weight
    return img_out.astype(np.float64)


# In[8]:

def main():
    # read degraded iamge (grayscale )
    s=window_stride
    orimg = cv2.imread("parrotinpainted.png")
    orimg=cv2.resize(orimg,(128,128))
    orimg=orimg[:,:,2]#[:,:,0] (b-channel); [:,:,1] (r-channel)
    image = cv2.imread("parrot_origin_01.png")
    image=image[:,:,2]#[:,:,0] (b-channel); [:,:,1] (r-channel)
    #orimg = cv2.resize(orimg,(342,341))
    #image = cv2.resize(image,(256,256))
    #image=image[0:254,:]

    r,c = image.shape
    print(r)
    print(c)
    mask=cv2.imread("parrot_mask_01.png",0)
    #glasses case
    #mask[mask<200]=1
    #mask[mask>=200]=0
    #parraot case, girls case
    mask[mask!=255]=0
    mask[mask==255]=1
    plt.imshow(mask*255)
    plt.show()
    #mask = np.ones((r,c))
    #mask[image!=orimg] = 0
    #mask[image==0] = 0
    #mask[abs(image-orimg)>10] = 0
    mask,_ = getImagePatches(mask,s)

    print('num_dict(设置稀疏向量的长度):',num_dict,'max_sparsity(稀疏向量最大稀疏数):',max_sparsity)
    # denose the given image
    inpainting_image = inpainting(image, dict_size=num_dict, sparsity=max_sparsity, mask=mask)

    #     noisy_psnr = 20*np.log10(np.amax(image)) - 10*np.log10(pow(np.linalg.norm(image - noisy_image), 2)/noisy_image.size)
    #     final_psnr = 20*np.log10(np.amax(image)) - 10*np.log10(pow(np.linalg.norm(image - denoised_image), 2)/denoised_image.size)
    # save images
    # cv2.imwrite("lena.png".split(".")[0]+str(sigma)+"noisy.jpg", noisy_image.astype('uint8'))
    cv2.imwrite("parrot_original.png".split(".")[0]+"inpainting_r.png", inpainting_image.astype('uint8'))# "inpainting_b", "inpainting_g"
    # cv2.imwrite("lena.png".split(".")[0]+str(sigma)+"difference.jpg", np.abs(noisy_image - denoised_image).astype('uint8'))
    noisy_psnr = 20*np.log10(np.amax(orimg)) - 10*np.log10(pow(np.linalg.norm(orimg - image), 2)/image.size)
    final_psnr = 20*np.log10(np.amax(orimg)) - 10*np.log10(pow(np.linalg.norm(orimg - inpainting_image), 2)/inpainting_image.size)
    print("PSRN: {}(noisy), {}(denoised))".format(noisy_psnr, final_psnr))
    #     print("PSRN: {}(noisy), {}(denoised))".format(noisy_psnr, final_psnr))
    return 0

if __name__=='__main__':
    main()


# In[9]:

plt.rcParams['figure.figsize'] = (10.0, 8.0)
a = cv2.imread("parrot_originalinpainting_r.png", 0)#"a2inpainting_b", "a2inpainting_g"
plt.imshow(a,"gray")


# In[10]:

b = cv2.imread("parrot_origin_01.png", 0)
plt.imshow(b,"gray")


# In[12]:

plt.rcParams['figure.figsize'] = [20, 10]
plt.subplot(1,2,1)
plt.title("inpainting")
plt.imshow(a,"gray")
plt.subplot(1,2,2)
plt.title("origin")
plt.imshow(b,"gray")
plt.show()
