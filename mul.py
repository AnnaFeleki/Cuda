#MATRIX MULTIPLICATION
from __future__ import division
from numba import cuda
import numpy
import math
from timeit import default_timer as time
import numpy as np





@cuda.jit
def matmul(A, B, C):

    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
   
def matmul_cpu(A, B, C):

  for i in range(len(A)):
    for j in range(len(B[0])):
      # iterate through rows of Y
      for k in range(len(B)):
        C[i][j] += A[i][k] * B[k][j]

   
# Other way
# from numba import cuda, float32

# # Controls threads per block and shared memory usage.
# # The computation will be done on blocks of TPBxTPB elements.
# TPB = 16

# @cuda.jit
# def fast_matmul(A, B, C):
#     # Define an array in the shared memory
#     # The size and type of the arrays must be known at compile time
#     sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
#     sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

#     x, y = cuda.grid(2)

#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     bpg = cuda.gridDim.x    # blocks per grid

#     if x >= C.shape[0] and y >= C.shape[1]:
#         # Quit if (x, y) is outside of valid C boundary
#         return

#     # Each thread computes one element in the result matrix.
#     # The dot product is chunked into dot products of TPB-long vectors.
#     tmp = 0.
#     for i in range(bpg):
#         # Preload data into shared memory
#         sA[tx, ty] = A[x, ty + i * TPB]
#         sB[tx, ty] = B[tx + i * TPB, y]

#         # Wait until all threads finish preloading
#         cuda.syncthreads()

#         # Computes partial product on the shared memory
#         for j in range(TPB):
#             tmp += sA[tx, j] * sB[j, ty]

#         # Wait until all threads finish computing
#         cuda.syncthreads()

#     C[x, y] = tmp

#GET SIZES OF ARRAYS
n = int(input("Give rows of a "))
l = int(input("Give columns of a "))
m = int(input("Give columns of n "))


# Initialize the data arrays
# A = np.random.randint(1,10,size=(n,l))
# B = np.random.randint(1,10,size=(l,m))

A = np.random.rand(n,m)
B= np.random.rand(n,m)

# Copy the arrays to the device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

# Allocate memory on the device for the result
C_global_mem = cuda.device_array((n, m))

# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)




s = time()
# Start the kernel 
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
e= time()

tcuda = e - s


# Copy the result back to the host
C = C_global_mem.copy_to_host()

ss = time()
# Start the kernel 
matmul_cpu(A, B, C)
ee= time()


tcpu = ee - ss

# print("\n\n")
# print(A)
# print("\n\n")
# print(B)
# print("\n\n")
# print(C)
# print("\n\n")


print('cuda: %f' % tcuda)
print('cpu: %f' % tcpu)

cudaFree(dA)
cudaFree(dB)
cudaFree(dC)
