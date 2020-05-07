#ADD MATRICES



from timeit import default_timer as time
import numpy as np
from numba import cuda
import math
import timeit
#print(cuda.gpus)


@cuda.jit
def cu_square_matrix_mul(A, B, C):

    row,col=cuda.grid(2)
    if row <A.shape[0] and col <A.shape[1]:
    	C[row,col] = A[row,col] + B[row,col]



def cu_square_matrix(A, B, C):

	# print(A+B)
	C = A + B


#GET SIZES OF ARRAYS
n = int(input("Give rows "))
m = int(input("Give columns "))

#FILL WITH RANDOM VALUES

# B = np.random.randint(1,10,size=(n,m))
# C = np.empty_like(A)

A = np.random.rand(n,m)
B= np.random.rand(n,m)

C = np.empty_like(A)
#PRINT ARRAYS
# print(A)
# print("****")
# print(B)
print("N = %d x %d" % (n, m))


#ALLOCATE MEMORY TO GPU
dA = cuda.to_device(A)
dB = cuda.to_device(B)
dC = cuda.to_device(C)



#A KERNEL
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(A.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)



s = time()
cu_square_matrix_mul[(blockspergrid, threadsperblock)](dA, dB, dC)
e = time()




tcuda = e - s

print('cuda: %f' % tcuda)



ss = time()
cu_square_matrix(A, B, C)
ee = time()

tcpu = ee - ss

print('cpu: %f' % tcpu)



#RETURN ARRAY TO HOST
C = dC.copy_to_host()

#print(C)
cudaFree(dA)
cudaFree(dB)
cudaFree(dC)