#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define TPB 1024

__global__ void rightNeighborSum(double *d_res, double *d_V, long int n){
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n - 1) return;

    d_res[idx] = d_V[idx] + d_V[idx + 1];
}

__global__ void middleNeighborSum(double *d_res, double *d_V, long int n){
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0 && idx >= n - 1) return;

    d_res[idx - 1] = d_V[idx - 1] + d_V[idx + 1];
}

void rNeighborLauncher(double *res, double *V, long int n){
	double *d_res, *d_V;

	cudaMalloc(&d_V, n * sizeof(double));
	cudaMalloc(&d_res, (n - 1) * sizeof(double));

	cudaMemcpy(d_V, V, n * sizeof(double), cudaMemcpyHostToDevice);

	rightNeighborSum<<<(n + TPB - 1) / TPB, TPB>>>(d_res, d_V, n);

	cudaMemcpy(res, d_res, (n - 1)*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_V);
	cudaFree(d_res);
	cudaDeviceReset();
}

void mNeighborLauncher(double *res, double *V, long int n){
	double *d_res, *d_V;

	cudaMalloc(&d_V, n * sizeof(double));
	cudaMalloc(&d_res, (n - 2) * sizeof(double));

	cudaMemcpy(d_V, V, n * sizeof(double), cudaMemcpyHostToDevice);

	middleNeighborSum<<<(n + TPB - 1) / TPB, TPB>>>(d_res, d_V, n);

	cudaMemcpy(res, d_res, (n - 2)*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_V);
	cudaFree(d_res);
	cudaDeviceReset();
}