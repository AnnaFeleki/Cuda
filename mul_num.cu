//MULTIPLY INTEGER WITH ARRAY
#include <stdio.h>
#include <stdlib.h>

#define N 100

__global__ void MatMulInt(int *a, int b, int *c,int ROW, int COLUMNS){
    
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * COLUMNS + ix;

    if (ix < ROW && iy < COLUMNS)
    {
        c[idx] = a[idx] * b ;
    }
}

int main(){
    int ROW, COLUMNS;
    int b;

    //GET SIZES OF ARRAYS
    printf("please give rows of the array: ");
    scanf("%d",&ROW);
    
    printf("please give columns of the array ");
    scanf("%d",&COLUMNS);

    printf("please give integer you want to multiply ");
    scanf("%d",&b);
    
   
    int a[ROW][COLUMNS];

    int c[ROW][COLUMNS];



    for(int i=0;i<ROW;i++)
      for(int j=0;j<COLUMNS;j++)
      {
        a[i][j]=rand()%5;
      }

    //DEFINE POINTERS FOR GPU
    int *dev_a,  *dev_c;

    //DEFINE SIZE
    int size = ((ROW* COLUMNS)*sizeof(int));

    //ALLOCATE MEMORY FOR POINTERS
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_c, size);

    //TRANSFER TO HOST
    cudaMemcpy(dev_a, a, size ,cudaMemcpyHostToDevice);

    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 grid((ROW + threadsPerBlock.x - 1) / threadsPerBlock.x, (COLUMNS + threadsPerBlock.y - 1) / threadsPerBlock.y);

//    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
//    dim3 dimGrid( ceil(int(N)/int(threadsPerBlock.x)), ceil(int(N)/int(threadsPerBlock.y)) );


    clock_t start = clock();
    MatMulInt<<<grid,threadsPerBlock>>>(dev_a, b, dev_c, ROW, COLUMNS);
    clock_t end = clock();



    //RETURN RESULT TO DEVICE
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);



    cudaFree(dev_a); 

    cudaFree(dev_c);


//    printf("\n");
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
//  
//    printf("%d ",b);
//
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

    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Xronos GPU %f\n\n",seconds);


    clock_t start_cpu = clock();
     for(int i=0;i<ROW;i++)
    {
        for (int j=0;j<COLUMNS;j++)
        {
            c[i][j]= a[i][j]  *b;
        }
    }

    clock_t end_cpu = clock();


    float secondss = (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("Xronos CPU %f",secondss);






    return 0;
}
