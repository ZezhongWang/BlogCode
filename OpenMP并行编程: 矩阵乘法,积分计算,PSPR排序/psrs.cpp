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
#include <assert.h>
// 编译方式 g++ psrs.cpp -o psrs -lbenchmark -lpthread -fopenmp -D__DEBUG__&& ./psrs
// ans:49999950

using std::string;

#ifdef __DEBUG__
#define DEBUG(format,...) printf(format "\n", ##__VA_ARGS__)
#else
#define DEBUG(format,...)
#endif

void printArr(int *arr, int length, string msg){
#ifdef __DEBUG__
    printf("%s\n", msg.c_str());
    for (int i = 0 ; i < length ; ++i){
        printf("%d\t", arr[i]);
    }
    printf("\n");
#endif
}

#define SIZE 1024*1024/4
int *data;

void Merge(int *arr, int left, int mid, int right){
    int len = right - left + 1;
    int *temp = new int[len];
    int lhead = left, rhead = mid+1;
    for (int i = 0; i < len; ++i) {
        if (lhead == mid+1){
            for (int j = rhead; j <= right; ++j) {
                temp[i++] = arr[j];
            }
            break;
        }
        else if (rhead == right+1){
            for (int j = lhead; j <= mid; ++j) {
                temp[i++] = arr[j];
            }
            break;
        }

        if (arr[lhead] > arr[rhead]){
            temp[i] = arr[rhead];
            ++rhead;
        }
        else{
            temp[i] = arr[lhead];
            ++lhead;
        }
    }
    for (int i = 0; i < len; ++i) {
        arr[left + i] = temp[i];
    }
    delete temp;
}

/*
 * 默认自小到大
 */
void MergeSort(int *arr, int left, int right){
    if (left == right ) return;

    int mid = (left + right)/2;

    MergeSort(arr, left, mid);
    MergeSort(arr, mid+1, right);
    Merge(arr, left, mid, right);
}

void PSRS(int *arr, int length, int thread_num){
    if(thread_num == 1){
        MergeSort(arr, 0, length-1);
        return ;
    }
    int id, sample_index = 0;
    int base = length / thread_num; // 每段个数
    int *regular_sample = new int[thread_num*thread_num];
    int *pivot = new int[thread_num-1];  // 主元
//    int **pos = new int[thread_num][thread_num-1];  // 每个线程主元划分之后的划分点
//    int (*pos)[10] = new int[thread_num][10];  // 每个线程主元划分之后的划分点,为第一个大于的值
    int **pos = new int* [thread_num];
    for (int i = 0; i < thread_num; ++i) {
        pos[i] = new int[thread_num-1];
    }
    int *temp_arr = new int[length]; // 全局交换中间
    int *chunk_size = new int[thread_num];
    omp_set_num_threads(thread_num);

#pragma omp parallel shared(arr, length, thread_num, base, regular_sample) private(id)
    {
        // 局部排序
        id = omp_get_thread_num();
        assert(id != thread_num);
        if (id == thread_num - 1) {
            MergeSort(arr, id * base, length-1);
        }
        else MergeSort(arr, id * base, (id + 1) * base - 1);

        // 正则采样
#pragma omp critical
        for (int j = 0; j < thread_num; ++j)
            regular_sample[sample_index++] = arr[j * base / thread_num + id * base];

// 同步所有线程
#pragma omp barrier
        assert(sample_index == thread_num * thread_num);
#pragma omp master
        {
            printArr(arr, length, "Local Sorted result：");
            MergeSort(regular_sample, 0, sample_index - 1);
            printArr(regular_sample, thread_num * thread_num, "Regular Sampling after sort：");
            // 选出主元
            for (int j = 0; j < thread_num - 1; ++j) {
                pivot[j] = regular_sample[(j + 1) * thread_num];
            }
            printArr(pivot, thread_num-1, "Select Pivot result:");
        }
#pragma omp barrier
        // 主元划分
        int left = id * base, right = (id + 1) * base - 1, pivot_index = 0;
        if (id == thread_num - 1) right = length - 1;

        for (int j = left; j <= right; ++j) {
            if (pivot_index == thread_num - 1) break;

            if (arr[j] > pivot[pivot_index]) {
                pos[id][pivot_index++] = j;
                continue;
            }
            while (j == right && pivot_index != thread_num - 1)
                pos[id][pivot_index++] = j;
        }

    }
    for (int k = 0; k < thread_num; ++k) {
        DEBUG("Thread id = %d: Pivot divide point", k);
        printArr(pos[k], thread_num-1, "");
    }

    // 全局交换
    assert(thread_num >= 2);
    int start_index = 0, cpleft, cprigth;
    // 复制每组数据左端
    for (int i = 0; i < thread_num; ++i) {
        cpleft = i*base;
        cprigth = pos[i][0] - 1;
        memcpy(temp_arr+start_index, arr + cpleft, sizeof(int)*(cprigth - cpleft + 1));
        start_index += cprigth - cpleft + 1;
    }
    chunk_size[0] = start_index;
    printArr(temp_arr, length, "Copy left end");
    // 每组数据中间部分
    // 遍历 1-thread_num-1 的pos
    for (int i = 1; i < thread_num-1; ++i) {
        // 遍历 0 - thread_num 组数据
        for (int j = 0; j < thread_num; ++j) {
            cpleft = pos[j][i-1];
            cprigth = pos[j][i] - 1;
            memcpy(temp_arr+start_index, arr + cpleft, sizeof(int)*(cprigth - cpleft + 1));
            start_index += cprigth - cpleft + 1;
        }
        chunk_size[i] = start_index;
    }
    printArr(temp_arr, length, "Copy mid");
    // 复制每组数据右端
    for (int i = 0; i < thread_num; ++i) {
        cpleft = pos[i][thread_num-2];
        cprigth = (i+1)*base -1;
        memcpy(temp_arr+start_index, arr + cpleft, sizeof(int)*(cprigth - cpleft + 1));
        start_index += cprigth - cpleft + 1;
    }
    chunk_size[thread_num-1] = start_index;
    printArr(temp_arr, length, "Copy right end");
    assert(start_index == length);

#pragma omp parallel shared(temp_arr, chunk_size) private(id)
    {
        id = omp_get_thread_num();
        assert(id != thread_num);
        if (id == 0)    MergeSort(temp_arr, 0, chunk_size[0]-1);
        else MergeSort(temp_arr, chunk_size[id-1], chunk_size[id]-1);
    }
#pragma omp barrier
    memcpy(arr, temp_arr, sizeof(int)*length);
    delete []temp_arr;
    delete []regular_sample;
    delete []pivot;
    delete []chunk_size;
    for (int i = 0; i < thread_num; ++i) {
        delete []pos[i];
    }
    printArr(arr, length, "*********PSRS Sort Final Result*********");
}



static void BM_PSRS(benchmark::State& state) {
    int *temp = new int[SIZE];
    while (state.KeepRunning()){
        memcpy(temp, data, sizeof(int)*SIZE);
        PSRS(temp, SIZE, state.range(0));
    }
    delete []temp;

}

BENCHMARK(BM_PSRS)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);//->Arg(10000)->Arg(50000);

int main(int argc, char **argv){
#ifndef _OPENMP
    fprintf(stderr, "OpenMP not supported");
#endif
    int arr1[] = {15, 46, 48, 93, 39, 6, 72, 91, 14, 36, 69, 40, 89,
                  61, 97, 12, 21, 54, 53, 97, 84, 58, 32, 27, 33, 72, 20};
    PSRS(arr1, 27, 3);
    int length = SIZE;
    data = new int[length];
    int fin = open("testArr_1G", O_RDONLY | O_CREAT);
    if(fin==-1)
    {
        printf("Cannot Open file\n");
        return -1;
    }
    read(fin, data, length*sizeof(int));
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
}

/*
 * 128M
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1     5575425 ns    5574730 ns        119
BM_PSRS/2     4537370 ns    4536497 ns        138
BM_PSRS/4     3564377 ns    3531409 ns        199
BM_PSRS/8     5804759 ns    1744159 ns        399
BM_PSRS/16    4048790 ns     812905 ns        846
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1     5780015 ns    5776708 ns        108
BM_PSRS/2     4591845 ns    4590313 ns        130
BM_PSRS/4     3643118 ns    3583664 ns        191
BM_PSRS/8     3925306 ns    1762464 ns        407
BM_PSRS/16    4054903 ns     807133 ns        846
 */

/*
 * 256M
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1    11740782 ns   11738878 ns         58
BM_PSRS/2     9483048 ns    9480814 ns         72
BM_PSRS/4     7420237 ns    7418210 ns         86
BM_PSRS/8    14766490 ns    3751146 ns        195
BM_PSRS/16    8504771 ns    1779683 ns        383
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1    11745879 ns   11742853 ns         59
BM_PSRS/2     9543409 ns    9541244 ns         70
BM_PSRS/4     7370735 ns    7335996 ns         83
BM_PSRS/8     7951400 ns    3607448 ns        196
BM_PSRS/16    8217029 ns    1765328 ns        395
 */

/*
 * 512M
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1    26766262 ns   26762375 ns         26
BM_PSRS/2    21546728 ns   21543054 ns         32
BM_PSRS/4    16343672 ns   16325361 ns         41
BM_PSRS/8    18270290 ns    7996515 ns         90
BM_PSRS/16   27815311 ns    4239498 ns        100
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1    24573469 ns   24568969 ns         28
BM_PSRS/2    23544930 ns   23522342 ns         34
BM_PSRS/4    16650627 ns   16642293 ns         37
BM_PSRS/8    22777775 ns    7315934 ns         93
BM_PSRS/16   16525077 ns    3674966 ns        188

 */
/*
 * 1G
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1    51594318 ns   51581710 ns         11
BM_PSRS/2    41460975 ns   41448931 ns         17
BM_PSRS/4    31809650 ns   31793747 ns         22
BM_PSRS/8    32284587 ns   15263166 ns         47
BM_PSRS/16   32827435 ns    7509254 ns         93
--------------------------------------------------
Benchmark           Time           CPU Iterations
--------------------------------------------------
BM_PSRS/1    51576864 ns   51568442 ns         12
BM_PSRS/2    53778408 ns   49312449 ns         11
BM_PSRS/4    32041725 ns   31869723 ns         21
BM_PSRS/8    33302379 ns   15176198 ns         47
BM_PSRS/16   38370519 ns    7830948 ns         92
 */