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

    # Define grid and block size
    rows, cols = A.shape
    threads_per_block = 1000
    blocks_per_grid = (rows * cols + threads_per_block - 1) // threads_per_block  # rows * cols == A.size

    # Flatten the input matrices for 1D indexing
    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = np.zeros_like(A_flat)  # C_flat = np.empty_like(A_flat)

    # Allocate device memory
    device_A = cuda.to_device(A_flat)
    device_B = cuda.to_device(B_flat)
    device_C = cuda.device_array_like(C_flat)  # device_C = cuda.to_device(C_flat)

    # Launch kernel
    max_kernel[blocks_per_grid, threads_per_block](device_A, device_B, device_C)

    # Copy the result back to the host
    C_flat = device_C.copy_to_host()

    # Reshape the result to the original matrix shape
    return C_flat.reshape(rows, cols)


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

    # Koren: The cuda.grid does not work:
    # tx = cuda.threadIdx.x  # Thread ID within a block
    # bx = cuda.blockIdx.x  # Block ID within the grid
    # bw = cuda.blockDim.x  # Number of threads per block
    # idx = tx + bx * bw  # Compute global thread index

    # Koren: The index is flattened
    # idx = cuda.grid(1)  # Get thread's global index

    # if idx < A.size:  # Ensure within bounds
    #     C[idx] = max(A.flat[idx], B.flat[idx])  # Koren: Maybe without flat: C[idx] = max(A[idx], B[idx])

    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        C[row, col] = max(A[row, col], B[row, col])


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
