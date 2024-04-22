/*
PROGRAM: Vector addition
HOW TO RUN :
$ nvcc -arch=sm_35 Exercise3.cu -o run_Exercise3
$ ./run_Exercise3
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 4

// Función Kernel que se ejecuta en el Device.
__global__ void Multiplica_Matrices_GM(float *C,float *A,float *B, int nfil, int ncol)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int index = idy*ncol + idx;
  if (idy<nfil && idx<ncol){
    float sum = 0.0f;
    for (int k=0; k<ncol; k++){
        sum += A[idy*ncol+k]*B[k*ncol+idx];
    }
	C[index] = sum;
  }
}

// Código principal que se ejecuta en el Host
int main(void){
	float *A_h,*B_h,*C_h; //Punteros a arreglos en el Host
	float *A_d,*B_d,*C_d;  //Punteros a arreglos en el Device
    const int nfil = 12;
    const int ncol = 12;
	const int N = nfil*ncol;  //Número de elementos en los arreglos  (probar 1000000)

    // GPU Time
    cudaEvent_t start, stop;
    float time;

	size_t size=N * sizeof(float);

	A_h = (float *)malloc(size); // Pedimos memoria en el Host
	B_h = (float *)malloc(size);
	C_h = (float *)malloc(size);//También se puede con cudaMallocHost

	//Inicializamos los arreglos A,B en el Host
	for (int i=0; i<nfil; i++){
		for (int j=0; j<ncol; j++){
            A_h[i*ncol+j] = 1.0f;
            B_h[i*ncol+j] = 2.0f;
        }
	}

	printf("\nMatriz a:\n");
	for (int i=0; i<N; i++) printf("%f ", A_h[i]);
	printf("\n\nMatriz b:\n");
	for (int i=0; i<N; i++) printf("%f ", B_h[i]);

	cudaMalloc((void **) &A_d,size);   // Pedimos memoria en el Device
	cudaMalloc((void **) &B_d,size);
	cudaMalloc((void **) &C_d,size);

	//Pasamos los arreglos a y b del Host al Device
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	//Realizamos el cálculo en el Device
	dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
    dim3 n_blocks(ceil (ncol/block_size.x),ceil(nfil/block_size.y));

    Multiplica_Matrices_GM<<<n_blocks,block_size>>>(C_d,A_d,B_d,nfil,ncol);

	//Pasamos el resultado del Device al Host
	cudaMemcpy(C_h, C_d, size,cudaMemcpyDeviceToHost);

	//Resultado
	printf("\n\nMatriz c:\n");
	for (int i=0; i<10; i++){
        for (int j=0; j<10; j++){
            printf("%.2f",C_h[i*ncol+j]);
        }
        printf("\n");
    }

	printf("\n\nFin del programa...\n");
	//system("pause");

	// Liberamos la memoria del Host
	free(A_h);
	free(B_h);
	free(C_h);

	// Liberamos la memoria del Device
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	return(0);
}