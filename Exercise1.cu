/*
PROGRAM: Hello world
HOW TO RUN :
$ nvcc -arch=sm_35 Exercise1.cu -o run_Exercise1
$ ./run_Exercise1
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloCuda(float e){
    printf("Hello, I am thread %d of block %d with value e=%f\n",threadIdx.x,blockIdx.x, e);
}

int main(int argc, char **argv){
    printf("Hello World\n");
    helloCuda<<<3,4>>>(2.71828f);

    cudaDeviceReset();
    return(0);
}
