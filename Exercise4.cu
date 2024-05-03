#include <stdio.h>
#include <cuda_runtime.h>
#define TPB 1024

__global__ void rightNeighborSum(double *d_res, double *d_V, long int n){
    long int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= n - 1) return;

    d_res[idx] = d_V[idx] + d_V[idx + 1];
}

double *generateRandomArray(int N);
double *generarS1_Seq(double *V, long int n);
double *rNeighborLauncher(double *V, long int n);

int main(int argc, char* argv[]){
	srand(time(NULL));
    double t_ini, t_fin, time_generateData, time_cpu_seconds, time_gpu_seconds, time_cpu2_seconds, time_gpu2_seconds;
    double *V, *S1_cpu, *S1_gpu, *S2_cpu, *S2_gpu;
    long int n=500000000;

    printf("De cuantos elementos es el vector? ");
    scanf("%ld",&n);

    t_ini=clock();
    V=generateRandomArray(n);
    t_fin=clock();
    time_generateData=(t_fin-t_ini)/CLOCKS_PER_SEC;

    t_ini=clock();
    S1_cpu=generarS1_Seq(V,n-1);
    t_fin=clock();
    time_cpu_seconds=(t_fin-t_ini)/CLOCKS_PER_SEC;

    t_ini=clock();
    S1_gpu=rNeighborLauncher(V,n);
    t_fin=clock();
    time_gpu_seconds=(t_fin-t_ini)/CLOCKS_PER_SEC;

    printf("Tiempo para generar datos: %lf segundos.\n",time_generateData);
    printf("Tiempo de procesamiento en CPU del vector S1: %lf segundos.\n",time_cpu_seconds);
    printf("Tiempo de procesamiento en GPU del vector S1: %lf segundos.\n",time_gpu_seconds);

    free(V);

    return 0;
}

double *generateRandomArray(int N){
    double *x;
    x=(double*)malloc(N*sizeof(double));
    for(int i=0; i<N; i++){
        x[i]=rand()%1000000+(double)(rand()%1000000)/1000000;
    }

    return x;
}

double *generarS1_Seq(double *V, long int n){
	double *x;
    x=(double*)malloc(n*sizeof(double));
    for(int i=0; i<n; i++){
    	x[i]=V[i]+V[i+1];
    }

    return x;
}

double *rNeighborLauncher(double *V, long int n){
	double *res, *d_res, *d_V;
    res=(double*)malloc(n*sizeof(double));

	cudaMalloc(&d_V,n*sizeof(double));
	cudaMalloc(&d_res,(n-1)*sizeof(double));

	cudaMemcpy(d_V,V,n*sizeof(double),cudaMemcpyHostToDevice);

	rightNeighborSum<<<(n+TPB-1)/TPB,TPB>>>(d_res,d_V,n);

	cudaMemcpy(res,d_res,(n-1)*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_V);
	cudaFree(d_res);
	cudaDeviceReset();

	return res;
}