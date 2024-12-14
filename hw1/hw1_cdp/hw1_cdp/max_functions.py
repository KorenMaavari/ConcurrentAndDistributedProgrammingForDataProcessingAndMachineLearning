import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def max_cpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """

    # return np.array([[max(a, b) for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)])
    rows, cols = A.shape
    # result = np.zeros((rows, cols), dtype=A.dtype)
    result = np.zeros_like(A)  # Koren: Is it a numpy vectorize operation?
    for i in range(rows):  # for i in range(A.shape[0]):
        for j in range(cols):  # for j in range(A.shape[1]):
            result[i, j] = max(A[i, j], B[i, j])  # result[i, j] = A[i, j] if A[i, j] > B[i, j] else B[i, j]
    return result


@njit(parallel=True)
def max_numba(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """

    rows, cols = A.shape
    # result = np.zeros((rows, cols), dtype=A.dtype)
    # result = np.empty_like(A)
    result = np.zeros_like(A)
    for i in prange(rows):  # for i in prange(A.shape[0]):
        for j in prange(cols):  # for j in range(A.shape[1]):
            result[i, j] = max(A[i, j], B[i, j])  # result[i, j] = A[i, j] if A[i, j] > B[i, j] else B[i, j]
    return result


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    #A and B are given matrixes of size(1000,1000) with integer values of range [0,255]
    # Define grid and block size
    threads_per_block = 1000
    blocks = 1000

    C = np.zeros_like(A)

    # Allocate device memory
    device_A = cuda.to_device(A)
    device_B = cuda.to_device(B)
    device_C = cuda.to_device(C)

    # Launch kernel
    max_kernel[blocks, threads_per_block](device_A, device_B, device_C)

    # Copy the result back to the host
    C = device_C.copy_to_host()

    return C


@cuda.jit
def max_kernel(A, B, C):
    """
    Koren:
    CUDA kernel for calculating the element-wise maximum of two matrices.
    Parameters:
        A (cuda.device_array): First input matrix (device array).
        B (cuda.device_array): Second input matrix (device array).
        C (cuda.device_array): Output matrix to store the element-wise maximum.
    """
    tx = cuda.threadIdx.x  # Thread ID within a block
    bx = cuda.blockIdx.x  # Block ID within the grid
    # We have 1000*1000 matrix.
    # We have total of 1000 * 1000 threads (1000 blocks, 1000 threads in a block).
    # Meaning: blockIdx will represent the matrix row, threadIdx will represent the matrix column.

    if tx < A.shape[0] and bx < A.shape[1]:
        C[bx, tx] = max(A[bx, tx], B[bx, tx])


def verify_solution():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    if not np.all(max_cpu(A, B) == np.maximum(A, B)):
        print('[-] max_cpu failed')
        exit(0)
    else:
        print('[+] max_cpu passed')

    if not np.all(max_numba(A, B) == np.maximum(A, B)):
        print('[-] max_numba failed')
        exit(0)
    else:
        print('[+] max_numba passed')

    if not np.all(max_gpu(A, B) == np.maximum(A, B)):
        print('[-] max_gpu failed')
        exit(0)
    else:
        print('[+] max_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    print('[*] CPU:', timer(max_cpu))
    print('[*] Numba:', timer(max_numba))
    print('[*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    verify_solution()
    max_comparison()
