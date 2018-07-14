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

// 编译方式 g++ calcTrap.cpp -o trap -lbenchmark -lpthread -fopenmp
// ans:49999950

double f(double x){
    return x;
}

double Trap(double a, double b, int n, double(*f)(double), int chunk){
    double integral, h;
    int k;
    h = (b-a)/n;
    integral = (f(a) + f(b))/2.0;

#pragma omp parallel for reduction(+:integral) schedule(dynamic, chunk)
    for (int k = 1; k <= n-1 ; ++k) {
        integral += f(a+k*h);
    }
    integral = integral*h;

    return integral;
}



static void BM_matrix_mul(benchmark::State& state) {

    while (state.KeepRunning())
        Trap(10.0, 10000.0, 100000, f, state.range(0));
}



BENCHMARK(BM_matrix_mul)->Arg(250)->Arg(500)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000);

int main(int argc, char **argv){
#ifndef _OPENMP
    fprintf(stderr, "OpenMP not supported");
#endif
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    printf("Ans = %lf", Trap(10.0, 10000.0, 100000, f, 10));
}


// 232000 ns 无并行
// 114849 ns reduction(+:integral)


/*
 *  schedule(static, chunk = 250, 500, 1000, 5000, 10000, 50000)
-----------------------------------------------------------
Benchmark                    Time           CPU Iterations
-----------------------------------------------------------
BM_matrix_mul/250       117050 ns     115962 ns       6183
BM_matrix_mul/500       127165 ns     122021 ns       6078
BM_matrix_mul/1000      115240 ns     112654 ns       5926
BM_matrix_mul/5000      112618 ns     111970 ns       5800
BM_matrix_mul/10000     119968 ns     119345 ns       5542
BM_matrix_mul/50000     148397 ns     147609 ns       4579

-----------------------------------------------------------
Benchmark                    Time           CPU Iterations
-----------------------------------------------------------
BM_matrix_mul/250       113316 ns     113245 ns       5917
BM_matrix_mul/500       114467 ns     114345 ns       5761
BM_matrix_mul/1000      116854 ns     115439 ns       5972
BM_matrix_mul/5000      113140 ns     112582 ns       5972
BM_matrix_mul/10000     132434 ns     127993 ns       5493
BM_matrix_mul/50000     207952 ns     204518 ns       3413
 */

/*
 *  schedule(dynamic, chunk = 250, 500, 1000, 5000, 10000, 50000)
-----------------------------------------------------------
Benchmark                    Time           CPU Iterations
-----------------------------------------------------------
BM_matrix_mul/250       115383 ns     115012 ns       6037
BM_matrix_mul/500       115422 ns     114665 ns       5843
BM_matrix_mul/1000      115055 ns     114825 ns       5719
BM_matrix_mul/5000      119371 ns     118095 ns       5827
BM_matrix_mul/10000     134629 ns     130171 ns       4945
BM_matrix_mul/50000     292827 ns     269824 ns       3579

-----------------------------------------------------------
Benchmark                    Time           CPU Iterations
-----------------------------------------------------------
BM_matrix_mul/250       115136 ns     112855 ns       6162
BM_matrix_mul/500       147566 ns     122738 ns       6356
BM_matrix_mul/1000      115024 ns     111260 ns       6200
BM_matrix_mul/5000      136719 ns     125605 ns       5313
BM_matrix_mul/10000     193029 ns     177285 ns       5621
BM_matrix_mul/50000     241088 ns     196514 ns       2885
 */



