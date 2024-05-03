#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 5

void printVector(double *Vector, int n);

int main(void){
    double *V, *S1_cpu, *S2_cpu, *S1_gpu, *S2_gpu;
    V = (double*)malloc(N * sizeof(double));
    S1_cpu = (double*)malloc((N-1) * sizeof(double));
    S1_gpu = (double*)malloc((N-1) * sizeof(double));
    S2_cpu = (double*)malloc((N-2) * sizeof(double));
    S2_gpu = (double*)malloc((N-2) * sizeof(double));

    srand(time(NULL));

    //Initialize input arrays
    for (int i = 0; i < N; i++){
        V[i] = rand() % 100;
    }
    printf("V vector:\n");
    printVector(V,N);

    //Ejercicio A: Serial CPU
    for (int i = 0; i < N - 1; i++){
        S1_cpu[i] = V[i] + V[i + 1];
    }
    printf("\ncpu result for s1: \n");
    printVector(S1_cpu,N - 1);
    
    //Ejercicio A: Parallel GPU
    rNeighborLauncher(S1_gpu, V, N);
    printf("\ngpu result for s1: \n");
    printVector(S1_gpu,N - 1);

    //Ejercicio B: Serial CPU
    for (int i = 1; i < N - 1; i++){
        S2_cpu[i - 1] = (V[i - 1] + V[i + 1])/2;
    }
    printf("\ncpu result for s2: \n");
    printVector(S2_cpu,N - 2);

    //Ejercicio B: Parallel GPU
    mNeighborLauncher(S2_gpu, V, N);
    printf("\ngpu result for s2: \n");
    printVector(S2_cpu,N - 2);

    free(V);
    free(S1_cpu);
    free(S1_gpu);
    free(S2_cpu);
    free(S2_gpu);
}

void printVector(double *Vector, int n){
    for (int i = 0; i < n; i++){
        printf("%.2f ", Vector[i]);
    }
    printf("\n");
}