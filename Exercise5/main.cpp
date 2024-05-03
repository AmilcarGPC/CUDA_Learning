#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 5
#define M 5

void printMatrix(double *Vector, int n, int m);

int main(void){
    double *A, *B, *C1_cpu, *C2_cpu, *C1_gpu, *C2_gpu;
    size_t size = N * M * sizeof(double);
    srand(time(NULL));

    A = (double*)malloc(size);
    B = (double*)malloc(size);
    C1_cpu = (double*)malloc(size);
    C1_gpu = (double*)malloc(size);
    C2_cpu = (double*)malloc(size);
    C2_gpu = (double*)malloc(size);

    const double alpha = 0.5;
    
    //Initialize input matrix
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            //A[i*M+j] = rand() % 100;
            A[i * M + j] = i * M + j;
        }
    }
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            //B[i*M+j] = rand() % 100;
            B[i * M + j] = N * M - i * M - j - 1;
        }
    }

    printf("A vector:\n");
    printMatrix(A,N,M);

    printf("B vector:\n");
    printMatrix(B,N,M);

    //Ejercicio A: Serial CPU
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            C1_cpu[i*M+j] = A[i*M+j] + B[(N-i-1)*M+(M-j-1)];
        }
    }
    printf("cpu result for c1: \n");
    printMatrix(C1_cpu,N,M);

    //Ejercicio A: Parallel GPU
    firstlastLauncher(C1_gpu,A,B,N,M);
    printf("gpu result for c1: \n");
    printMatrix(C1_gpu,N,M);

    //Ejercicio B: Serial CPU
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            C2_cpu[i*M+j] = alpha*A[i*M+j] + (1-alpha)*B[i*M+j];
        }
    }
    printf("cpu result for c2: \n");
    printMatrix(C2_cpu,N,M);

    //Ejercicio B: Parallel GPU
    weightedMatrixLauncher(C2_gpu,A,B,alpha,N,M);
    printf("gpu result for c2: \n");
    printMatrix(C2_gpu,N,M);

    free(A);
    free(B);
    free(C1_cpu);
    free(C1_gpu);
    free(C2_cpu);
    free(C2_gpu);
}

void printMatrix(double *Vector, int n, int m){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            printf("%.2f ", Vector[i*m+j]);
        }
        printf("\n");
    }
    printf("\n");
}