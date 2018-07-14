//
// Created by wesley on 18-4-21.
//

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <malloc.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <benchmark/benchmark.h>

#define N 500
#define M 500

// 编译方式 g++ matrixMul.cpp -o matrix -lbenchmark -lpthread -fopenmp


int *A, *y;
int *x;
int m, n;

void Mat_vect_mult(
                    int A[],
                    int x[],
                    int y[],
                    const int &m,
                    const int &n,
                    int chunk
        ){
    int i, j;
#pragma omp parallel for shared(A, x, y, m, n) private(i)
    for (i = 0; i < m; ++i) {
        y[i] = 0;
#pragma omp parallel for schedule(dynamic, chunk)
        for (j = 0; j < n; ++j) {
            y[i] += A[i*n + j]*x[j];
        }
    }
}

void init(){
    m = M;
    n = N;
    int fin = open("testData", O_RDONLY | O_CREAT);
    A = (int *)malloc(sizeof(int) * m * n);
    x = (int *)malloc(sizeof(int) * n);
    y = (int *)malloc(sizeof(int) * m);
    // 初始化A
    read(fin, A, m * n *sizeof(int));
    // 初始化X
    for (int i = 0; i < n; ++i) {
        x[i] = 1;
    }
//    for (int i = 0; i < m; ++i) {
//        for (int j = 0; j < n; ++j) {
//            printf("%d\t", A[i*n + j]);
//        }
//        printf("\n");
//    }
}


static void BM_matrix_mul(benchmark::State& state) {

    while (state.KeepRunning())
        Mat_vect_mult(A, x, y, m, n, state.range(0));
}



BENCHMARK(BM_matrix_mul)->Arg(10)->Arg(50)->Arg(100)->Arg(150)->Arg(200)->Arg(250);

int main(int argc, char **argv){
    init();
    #ifndef _OPENMP
        fprintf(stderr, "OpenMP not supported");
    #endif
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    for (int i = 0; i < m; ++i) {
        printf("%d\t", y[i]);
    }
}

// 675348 ns
// 432936 ns    加一个
// 423909 ns    加两个
// 446613 ns    for reduction(+:y[i])
// 510283 ns    schedule(static, 100)
// 424587 ns    schedule(static, 10)
// 458677 ns    schedule(static, 4)
// 546860 ns    schedule(dynamic, 10)

/*
 *  schedule(static, chunk = 10, 50, 100, 150)
---------------------------------------------------------
Benchmark                  Time           CPU Iterations
---------------------------------------------------------
BM_matrix_mul/10      442239 ns     424834 ns       1232
BM_matrix_mul/50      439412 ns     414957 ns       1236
BM_matrix_mul/100     447353 ns     446049 ns       1632
BM_matrix_mul/150     445679 ns     429528 ns       1597

---------------------------------------------------------
Benchmark                  Time           CPU Iterations
---------------------------------------------------------
BM_matrix_mul/10      443270 ns     436353 ns       1609
BM_matrix_mul/50      442818 ns     426392 ns       1630
BM_matrix_mul/100     451630 ns     434484 ns       1668
BM_matrix_mul/150     445859 ns     430546 ns       1354
 */

/*
 *  schedule(dynamic, chunk = 10, 50, 100, 150, 200, 250)
---------------------------------------------------------
Benchmark                  Time           CPU Iterations
---------------------------------------------------------
BM_matrix_mul/10      439406 ns     437708 ns       1613
BM_matrix_mul/50      400981 ns     400086 ns       1339
BM_matrix_mul/100     401700 ns     401126 ns       1343
BM_matrix_mul/150     402814 ns     400483 ns       1770
BM_matrix_mul/200     403648 ns     402614 ns       1778
BM_matrix_mul/250     398525 ns     397391 ns       1778

---------------------------------------------------------
Benchmark                  Time           CPU Iterations
---------------------------------------------------------
BM_matrix_mul/10      457178 ns     441894 ns       1271
BM_matrix_mul/50      446074 ns     432192 ns       1345
BM_matrix_mul/100     402164 ns     399626 ns       1748
BM_matrix_mul/150     397401 ns     395255 ns       1763
BM_matrix_mul/200     402396 ns     399089 ns       1776
BM_matrix_mul/250     400584 ns     398822 ns       1777

 */