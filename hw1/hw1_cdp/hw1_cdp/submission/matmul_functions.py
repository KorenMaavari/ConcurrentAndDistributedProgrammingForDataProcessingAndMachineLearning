import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    rows, cols = X.shape
    res_rows = rows
    res_cols = rows
    # res- the result of X(X)t.
    res = np.zeros((res_rows, res_cols))

    for i in range(rows):
        for j in range(rows):
            for k in range(cols):
                # note: X[j,k] =  Xtraspose[k,j]
                res[i, j] += X[i, k] * X[j, k]

    return res

@njit(parallel=True)
def matmul_transpose_numba(X):
    rows, cols = X.shape
    res_rows = rows
    res_cols = rows
    res = np.zeros((res_rows, res_cols))

    for i in prange(rows):
        for j in prange(rows):
            for k in prange(cols):
                res[i, j] += X[i, k] * X[j, k]

    return res


def matmul_transpose_gpu(X):
    # Define grid and block size
    threads_per_block = 1024
    blocks = 1

    rows, cols = X.shape
    res_rows = rows
    res_cols = rows
    # res- the result of X(X)t.
    res = np.zeros((res_rows, res_cols))

    # Allocate device memory
    device_X = cuda.to_device(X)
    device_res = cuda.to_device(res)

    # Launch kernel
    matmul_kernel[blocks, threads_per_block](device_X, device_res)

    # Copy the result back to the host
    res = device_res.copy_to_host()

    return res


@cuda.jit
def matmul_kernel(A, C):
    # note: we only have 1024 threads!
    # then - each thread will handle a part of the elements in the result matrix!

    threadIdx = cuda.threadIdx.x
    total_threads = cuda.blockDim.x
    rows, cols = A.shape

    # Compute how many elements each thread should handle
    total_elements = rows * rows  # Total number of elements in the result matrix
    elems_per_thread = (total_elements + total_threads - 1) // total_threads  # Divide work among threads

    for i in range(elems_per_thread):
        element_id = threadIdx + i * total_threads
        if element_id < total_elements:  # Ensure we're within bounds
            # Map element_id to a specific (row, col) in the result matrix
            row = element_id // rows
            col = element_id % rows

            # Compute the value of C[row, col]
            temp = 0
            for k in range(cols):
                temp += A[row, k] * A[col, k]

            C[row, col] = temp


def verify_solution():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    if not np.allclose(matmul_transpose_trivial(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_trivial failed')
        exit(0)
    else:
        print('[+] matmul_transpose_trivial passed')

    if not np.allclose(matmul_transpose_numba(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_numba failed')
        exit(0)
    else:
        print('[+] matmul_transpose_numba passed')

    if not np.allclose(matmul_transpose_gpu(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_gpu failed')
        exit(0)
    else:
        print('[+] matmul_transpose_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    verify_solution()
    matmul_comparison()
