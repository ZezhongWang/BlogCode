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
#include <mpi.h>
#include <benchmark/benchmark.h>

#define N 512   // 列
#define M 512   // 行

// 编译方式 g++ matrixMul.cpp -o matrix -lbenchmark -lpthread -fopenmp

//
//int *A, *y;
//int *x;
//int m, n;

void Mat_vect_mult(
                    int A[],
                    int x[],
                    int y[],
                    const int &m,
                    const int &n
        ){
    int i, j;
    for (i = 0; i < m; ++i) {
        y[i] = 0;
        for (j = 0; j < n; ++j) {
            y[i] += A[i*n + j]*x[j];
        }
    }
}

void Solve(){

    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int m = M, n = N;
    int *A, *y;

    if(rank == 0){
        int fin = open("testData", O_RDONLY | O_CREAT);
        A = (int *)malloc(sizeof(int) * m * n);
        read(fin, A, m * n *sizeof(int));
    }

    int *subMat = (int *)malloc(sizeof(int) * m * n / size);
    int *local_y = (int*)malloc(m/size * sizeof(int));
    int *x = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        x[i] = 1;
    }

    // 把矩阵各个列散射
    MPI_Scatter(A, m/size*n, MPI_INT, subMat, m/size*n, MPI_INT, 0, MPI_COMM_WORLD);


    Mat_vect_mult(subMat, x, local_y, m/size, n);

    if(rank == 0){
        y = (int *)malloc(sizeof(int) * m);
    }

    MPI_Gather(local_y, M/size, MPI_INT, y, M/size, MPI_INT, 0, MPI_COMM_WORLD);

//    if(rank == 0){
//        printf("Final Result:[");
//        int i;
//        for(i = 0 ; i < M-1 ; i++)
//            printf("%d,",y[i]);
//        printf("%d]\n",y[i]);
//    }

    free(local_y);
    free(subMat);
    free(x);
    if(rank == 0){
        free(y);
        free(A);
    }

    return ;
}

void init(){


}


static void BM_matrix_mul(benchmark::State& state) {
    while (state.KeepRunning()){
        Solve();
    }
}

BENCHMARK(BM_matrix_mul);

int main(int argc, char **argv){
    init();

    MPI_Init(NULL,NULL);
    Solve();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();

    MPI_Finalize();

}



// mpirun -np 1 ./matrixMul
//  1340320 ns  无并行


// mpirun -np 2 ./matrixMul
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul     561869 ns     561557 ns       1104

 */

// mpirun -np 4 ./matrixMul
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul     701463 ns     685804 ns        731

 */

// mpirun -np 8 ./matrixMul
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul   32374223 ns   17106772 ns         32
 */

// mpirun -np 16 ./matrixMul
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul   77658449 ns   19174935 ns         41
 */

