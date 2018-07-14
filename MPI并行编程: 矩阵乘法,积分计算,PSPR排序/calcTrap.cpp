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
#include <mpi.h>


// 编译方式 g++ calcTrap.cpp -o trap -lbenchmark -lpthread -fopenmp
// ans:49999950

double f(double x){
    return x;
}




double Trap(double a, double b, int n, double(*f)(double)){

    double integral, h;
    h = (b-a)/n;

    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int local_n;
    double local_a, local_b, local_sum, total_sum;

    local_n = n/size; // n必须为size的整数倍
    local_a = a + rank*local_n*h;
    local_b = local_a + local_n*h;

    for (int k = 0; k <= local_n - 1; ++k) {
        local_sum += f(local_a + k*h);
    }

    local_sum *= h;
//    printf("process %d : local_sum = %fl", rank, local_sum);

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

//    if(rank == 0)
    return total_sum;

}

//
//
static void BM_matrix_mul(benchmark::State& state) {

    while (state.KeepRunning())
        Trap(0, 10000.0, 131072, f);
}


BENCHMARK(BM_matrix_mul);

int main(int argc, char **argv){

    MPI_Init(NULL,NULL);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();

//    printf("Ans = %lf\n", Trap(0, 10, 100000, f));

    MPI_Finalize();


}


// 232000 ns 无并行
// 114849 ns reduction(+:integral)

// mpirun -np 1 ./calcTrap
//  304702 ns 无并行


// mpirun -np 2 ./calcTrap
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul     152960 ns     152864 ns       4375
 */

// mpirun -np 4 ./calcTrap
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul     138721 ns     135284 ns       3929
 */



// mpirun -np 8 ./calcTrap
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul     161147 ns      82932 ns      10475
 */

// mpirun -np 16 ./calcTrap
/*
-----------------------------------------------------
Benchmark              Time           CPU Iterations
-----------------------------------------------------
BM_matrix_mul     280238 ns      55820 ns      12618
 */

