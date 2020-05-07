//ADD TWO MATRICES
#include <stdio.h>
#include <stdlib.h>


__global__ void MatAdd(int *a, int *b, int *c, int ROW, int COLUMNS){
    
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * COLUMNS + ix;

    if (ix < ROW && iy < COLUMNS)
    {
        c[idx] = a[idx] + b[idx];
    }
}



int main(){

    //GET SIZES OF ARRAYS
    int ROW, COLUMNS;
    printf("please give rows of the array: ");
    scanf("%d",&ROW);
    
    printf("please give columns of the array ");
    scanf("%d",&COLUMNS);
    
   
    int a[ROW][COLUMNS];
    int b[ROW][COLUMNS];
    int c[ROW][COLUMNS];


    //FILL WITH RANDOM VALUES
    for(int i=0;i<ROW;i++)
      for(int j=0;j<COLUMNS;j++)
      {
        a[i][j]=rand()%5;
        b[i][j]=rand()%5;
      }

    //DEFINE POINTERS FOR GPU
    int *dev_a, *dev_b, *dev_c;

    //DEFINE SIZE
    int size = ((ROW* COLUMNS)*sizeof(int));

    //ALLOCATE MEMORY FOR POINTERS
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    //TRANSFER TO HOST
    cudaMemcpy(dev_a, a, size ,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);



    
    /*Do something*/
    
    

    dim3 threadsPerBlock(16, 16);
    dim3 grid((ROW + threadsPerBlock.x - 1) / threadsPerBlock.x, (COLUMNS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    

//    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
//    dim3 dimGrid( ceil(int(N)/int(threadsPerBlock.x)), ceil(int(N)/int(threadsPerBlock.y)) );



    clock_t start = clock();
    MatAdd<<<grid,threadsPerBlock>>>(dev_a, dev_b, dev_c, ROW, COLUMNS);
    clock_t end = clock();


    //RETURN RESULT TO DEVICE
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);



    cudaFree(dev_a); 
    cudaFree(dev_b); 
    cudaFree(dev_c);


    printf("\n");
//
//    for(int i=0;i<ROW;i++)
//    {
//      for(int j=0;j<COLUMNS;j++)
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
//    for(int i=0;i<ROW;i++)  
//    {
//      for(int j=0;j<COLUMNS;j++)
//      {
//        printf("%d ",b[i][j]);
//
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
//

    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Xronos GPU %f\n\n",seconds);


    clock_t start_cpu = clock();
     for(int i=0;i<ROW;i++)
    {
        for (int j=0;j<COLUMNS;j++)
        {
            c[i][j]= a[i][j]  + b[i][j];
        }
    }

    clock_t end_cpu = clock();


    float secondss = (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("Xronos CPU %f",secondss);


    

    return 0;
}
