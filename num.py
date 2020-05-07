#MULTIPLY INTEGER WITH ARRAY
from timeit import default_timer as time
import numpy as np
from numba import cuda
import math

# print(cuda.gpus)


@cuda.jit
def cu_square_matrix_mul(A, b, C):

    row,col=cuda.grid(2)
    if row <A.shape[0] and col <A.shape[1]:
    	C[row,col] = A[row,col] *b



def cu_square_matrix(A, b, C):

	C = A *b
	# print(C)


#GET SIZES OF ARRAYS
n = int(input("Give rows of a "))
m = int(input("Give columns of a "))
b = int(input("Give number to multiply "))

#FILL WITH RANDOM VALUES
# A = np.random.randint(1,10,size=(n,m))


A = np.random.rand(n,m)
B= np.random.rand(n,m)

C = np.empty_like(A)

#PRINT ARRAYS
# print(A)
# print("****")
# print(b)
print("N = %d x %d" % (n, m))



#ALLOCATE MEMORY TO GPU
dA = cuda.to_device(A)

dC = cuda.to_device(C)


#A KERNEL
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(A.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)





s = time()
cu_square_matrix_mul[(blockspergrid, threadsperblock)](dA, b, dC)
e = time()

#RETURN ARRAY TO HOST
C = dC.copy_to_host()

# print(C)


tcuda = e - s

print('cuda: %f' % tcuda)

ss = time()
cu_square_matrix(A,b,C)
ee = time()


tcpu = ee - ss

print('cpu: %f' % tcpu)

cudaFree(dA)
cudaFree(dC)