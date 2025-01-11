#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
# import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import numpy as np
from numba import cuda, prange
from numba import njit
import imageio
import matplotlib.pyplot as plt
import math

def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''

    # res- the result of correlation calculation between image and kernel.
    img_rows, img_cols = image.shape
    res = np.zeros((img_rows, img_cols))

    #allocate  everything on gpu memory
    device_img = cuda.to_device(image)
    device_ker = cuda.to_device(kernel)
    device_res = cuda.to_device(res)

    threads_per_block = (16, 16)  # 16x16 threads per block
    blocks_per_grid = (
        math.ceil(img_rows / threads_per_block[0]),
        math.ceil(img_cols / threads_per_block[1])
    )

    # Launch kernel. Blocks = img_rows, Threads_per_block = img_cols
    correlation_kernel[threads_per_block, blocks_per_grid](device_img, device_ker, device_res)

    # Copy the result back to the host
    res = device_res.copy_to_host()

    return res

@cuda.jit
def correlation_kernel(image, kernel, output):
    ''' Handles one cell of the image.

    Parameters
    ----------
    image: numpy array located on GPU.
    kernel: numpy array located on GPU.
    output: the output matrix, located on GPU as well.

    Returns
    -------
    void

    '''
    rows, cols = kernel.shape
    img_rows, img_cols = image.shape
    x_padd = rows // 2
    y_padd = cols // 2
    thread_x = cuda.threadIdx.x
    thread_y = cuda.threadIdx.y

    # get the index of the cell we are working on
    x = thread_x + cuda.blockIdx.x * cuda.blockDim.x
    y = thread_y + cuda.blockIdx.y * cuda.blockDim.y

    #check if it is valid
    if(x >= img_rows or y >= img_cols):
        return

    #initialize a local variable to write to, instead of writing to global mem. each iteration
    res = 0.0

    #calculate the correlation
    for row in range(rows):
        for col in range(cols):
            #check if the index is valid in the image.
            # if not - we won't do any addition, which equals to multiplying by 0 and adding the result
            idx_x = x - x_padd + row
            idx_y = y - y_padd + col
            if (idx_x >= 0 and idx_x < img_rows and idx_y >= 0 and idx_y < img_cols):
               res  += image[idx_x, idx_y] * kernel[row, col]

    # now, preform one write to the global memory.
    output[x, y] = res



@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''
    # first- padd the matrix with zeros so the correlation computation woll always be in bound.
    rows, cols = kernel.shape
    x_padd = rows // 2
    y_padd = cols // 2

    # result - the result of correlation calculation between image and kernel.
    img_rows, img_cols = image.shape
    result = np.zeros((img_rows, img_cols))

    for i in prange(img_rows):
        for j in prange(img_cols):

            # a local variable to calculate into
            res = 0.0
            for offset_x in prange(rows):
                for offset_y in prange(cols):
                    indx_x = i + offset_x - x_padd
                    indx_y = j + offset_y - y_padd
                    if(indx_x >= 0 and indx_y >= 0 and indx_x < img_rows and indx_y < img_cols):
                        res += kernel[offset_x, offset_y] * image[indx_x, indx_y]
            # now add the local calculation to memory
            result[i, j] = res

    return result


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    # your calculations
    pic = load_image()
    sobel_filter_submission = np.array([[1, 0, -1],
                                        [2, 0, -2],
                                        [1, 0, -1]])

    # sobel_filter1 = np.array([[3, 0, -3],
    #                           [10, 0, -10],
    #                           [3, 0, -3]])

    # sobel_filter2 = np.array([[1, 0, -1],
    #                           [2, 0, -1],
    #                           [1, 0, -2],
    #                           [2, 0, -2],
    #                           [1, 0, -1]])

    # sobel_filter3 = np.array([[1, 1, 1],
    #                           [1, 0, 1],
    #                           [1, 1, 1]])

    Gx = correlation_numba(sobel_filter_submission, pic)
    Gy = correlation_numba(np.transpose(sobel_filter_submission), pic)
    return np.sqrt((Gx ** 2) + (Gy ** 2))


def load_image():
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()

