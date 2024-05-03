#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define TPB 32

__global__ void firstLastSum(double *d_res, double *d_A, double *d_B, int n, int m){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx >= n || idy >= m) return;

    d_res[idx*m+idy] = d_A[idx*m+idy] + d_B[(n-idx-1)*m+m-idy-1];
}

__global__ void weightedMatrixSum(double *d_res, double *d_A, double *d_B, double alpha, int n, int m){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx >= n || idy >= m) return;

	d_res[idx*m+idy] = alpha*d_A[idx*m+idy] + (1-alpha)*d_B[idx*m+idy];
}

void firstlastLauncher(double *res, double *A, double *B, int n, int m){
	double *d_res, *d_A, *d_B;
	size_t size = n * m * sizeof(double);

	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_res,size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	//Realizamos el cálculo en el Device
	dim3 block_size(TPB, TPB);
    dim3 n_blocks(ceil((double)n/block_size.x), ceil((double)m/block_size.y));

	firstLastSum<<<n_blocks, block_size>>>(d_res, d_A, d_B, n, m);

	cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);
	cudaDeviceReset();
}

void weightedMatrixLauncher(double *res, double *A, double *B, double alpha, int n, int m){
	double *d_res, *d_A, *d_B;
	size_t size = n * m * sizeof(double);

	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_res,size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	//Realizamos el cálculo en el Device
	dim3 block_size(TPB, TPB);
    dim3 n_blocks(ceil((double)n/block_size.x), ceil((double)m/block_size.y));

	weightedMatrixSum<<<n_blocks, block_size>>>(d_res, d_A, d_B, alpha, n, m);

	cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);
	cudaDeviceReset();
}