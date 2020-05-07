//MATRIX MULTIPLICATION
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>

__global__ void matrixMul(int *a, int *b, int *c, int ROW, int COLUMNS, int temp)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < COLUMNS && row < ROW) 
    {
        for(int i = 0; i < temp; i++) 
        {
            sum += a[row * temp + i] * b[i * COLUMNS + col];
        }
        c[row * COLUMNS + col] = sum;
    }
    

}

int main()
{
   
    int ROW, COLUMNS;
    int temp;
    

    //GET SIZES OF ARRAYS
    printf("please give rows of A: ");
    scanf("%d",&ROW);
    
    printf("please give columns of A ");
    scanf("%d",&temp);

    printf("please give columns of B ");
    scanf("%d",&COLUMNS);
    
   
    int a[ROW][temp];
    int b[temp][COLUMNS];
    int c[ROW][COLUMNS];



    //FILL WITH RANDOM VALUES
    for(int i=0;i<ROW;i++)
      for(int j=0;j<temp;j++)
      {
        a[i][j]=rand()%5;
      }

     for(int i=0;i<temp;i++)
      for(int j=0;j<COLUMNS;j++)
      {
        b[i][j]=rand()%5;
      }


    //DEFINE POINTERS FOR GPU
    int *dev_a, *dev_b, *dev_c;


    //DEFINE SIZE
    int size_a = ((ROW * temp)*sizeof(int));
    int size_b = ((temp * COLUMNS)*sizeof(int));
    int size_c = ((ROW * COLUMNS)*sizeof(int));

    //ALLOCATE MEMORY FOR POINTERS
    cudaMalloc((void**)&dev_a, size_a);
    cudaMalloc((void**)&dev_b, size_b);
    cudaMalloc((void**)&dev_c, size_c);

    //TRANSFER TO HOST
    cudaMemcpy(dev_a, a, size_a ,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size_c, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 grid((ROW + threadsPerBlock.x - 1) / threadsPerBlock.x, (COLUMNS + threadsPerBlock.y - 1) / threadsPerBlock.y);

//    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
//    dim3 dimGrid( ceil(int(N)/int(threadsPerBlock.x)), ceil(int(N)/int(threadsPerBlock.y)) );


    clock_t start = clock();
    matrixMul<<<grid,threadsPerBlock>>>(dev_a, dev_b, dev_c, ROW, COLUMNS, temp);
    clock_t end = clock();


    //RETURN RESULT TO DEVICE
    cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost);



    cudaFree(dev_a); 
    cudaFree(dev_b); 
    cudaFree(dev_c);


    printf("\n");
//
//    for(int i=0;i<ROW;i++)
//    {
//      for(int j=0;j<temp;j++)
//      {
//        printf("%d ",a[i][j]);
//      }
//      printf("\n");
//    }
//
//
//    printf("\n\n\n");
//
//
//    for(int i=0;i<temp;i++)
//    {
//      for(int j=0;j<COLUMNS;j++)
//      {
//        printf("%d ",b[i][j]);
//      }
//      printf("\n");
//    }
//
//    printf("\n\n\n");
//
//
//
//
//    for(int i=0;i<ROW;i++)
//    {
//      for(int j=0;j<COLUMNS;j++)
//      {
//        printf("%d ",c[i][j]);
//
//      }
//      printf("\n");
//    }
//
    
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Xronos GPU %f\n\n",seconds);


    clock_t start_cpu = clock();
     //Carrying out matrix multiplication operation
    int i, j, k; 
        for (i = 0; i < ROW; i++) 
        { 
            for (j = 0; j < COLUMNS; j++) 
            { 
                c[i][j] = 0; 
                for (k = 0; k < temp; k++) 
                    c[i][j] += a[i][k]*b[k][j]; 
            } 
        } 

    clock_t end_cpu = clock();


    float secondss = (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("Xronos CPU %f",secondss);



    return 0;
}
