//
// Created by w2w on 18-7-5.
//

#include "mpi.h"
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "digitcount.h"
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <cstdint>
using namespace std;

const long long MEMORY = (long long)4*1024*1024;    //设置用来存储读入数据的内存大小
unsigned int ARRAY_SIZE = UINT32_MAX;
int THREAD_NUM = 8;
#define MAPNUM_SEND_TAG 666
#define NUMCNT_SEND_TAG 667
#define PRINT_RESULT_LIMIT 25

// for testing
void printMap(map<int, long> &m, string msg){
#ifdef __DEBUG__
    printf("%s\n", msg.c_str());
    int i = 0;
    for(map<int, long>::iterator iter = m.begin();
            iter != m.end(); iter ++){
        ++i;
        if(i>PRINT_RESULT_LIMIT)    break;
        printf("%d:%ld;\t", iter->first, iter->second);
    }

    printf("\n");
#endif
}

// for testing
void printArr(int *arr, int length, string msg){
#ifdef __DEBUG__
    printf("%s\n", msg.c_str());
    for (int i = 0 ; i < length ; ++i)
        printf("%d\t", arr[i]);
    printf("\n");
#endif
}


unsigned int getIndex(const int &val){
    return *((unsigned int*)&val);
}

int getValue(const unsigned int &idx){
    return *((int *)&idx);
}

void TEST_printResult(short *arr){
    printf("Part of Result Array:(only display %d of the result\n", PRINT_RESULT_LIMIT);
    for (int i = 0; i < PRINT_RESULT_LIMIT; ++i) {
        printf("%d:%d;\t", i, arr[getIndex(i)]);
    }
    printf("\n");
}


#ifdef __DEBUG__
#define PRINT printf
#else
#define PRINT
#endif

DigitCount::DigitCount(string inFile, long memoryCanBeUsed):
    _inFile(inFile), _memoryCanBeUsed(memoryCanBeUsed),
    _fin(NULL), _FileSize(0)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_size);
    _fin = fopen(inFile.c_str(), "r");
    struct stat fileStatus;
    stat(inFile.c_str(), &fileStatus);
    _FileSize = fileStatus.st_size;
    if(_rank == 0)
        PRINT("FileSize = %lld\n", _FileSize);
    defineTransportStruct();
}

void DigitCount::defineTransportStruct()
{

    MPI_Datatype oldtypes[2];
    int          blockcounts[2];
    // MPI_Aint type used to be consistent with syntax of
    // MPI_Type_extent routine
    MPI_Aint    offsets[2], extent;

    // setup description of the 1 MPI_INT fields num
    offsets[0] = 0;
    oldtypes[0] = MPI_LONG;
    blockcounts[0] = 1;
    // setup description of the 1 MPI_LONG fields cnt
    MPI_Type_extent(MPI_LONG, &extent);
    offsets[1] = extent;
    oldtypes[1] = MPI_INT;
    blockcounts[1] = 1;

    // define structured type and commit it
    MPI_Type_struct(2, blockcounts, offsets, oldtypes, &MPI_NumCnt);
    MPI_Type_commit(&MPI_NumCnt);
}



void DigitCount::distributeTask(map<int, long> &digit2num) {
    // 除2是因为每份数据因为分了master和slave的原因,会有两份
    int readFileTimes = ceil((double)2 * _FileSize / _memoryCanBeUsed);

    if(_rank == 0)
        PRINT("readFileTimes = %d\n", readFileTimes);
    // 文件的int的数量
    int FileDigitNum = _FileSize / sizeof(int);
    // 主进程开内存
    if(_rank == 0)
        _arrayInMem = new int[_memoryCanBeUsed / 2 / sizeof(int)];
    // 所有进程开内存
    int *procDigit = new int[_memoryCanBeUsed / 2 / sizeof(int) / _size];
    // 直接使用Map来进行计数
    map<int, long> num2cnt;
    double count_duration = 0;
    for (int i = 0; i < readFileTimes; ++i) {
        // 每次主进程读取的digitNum
        int digitNum = min((int)(_memoryCanBeUsed / 2 / sizeof(int)), FileDigitNum - digitNum*i);
        // 每个进程对应的数量
        int procDigitNum = digitNum / _size;
        // 主进程读取数据
        if(_rank == 0){
            fseek(_fin, _memoryCanBeUsed / 2 * i, SEEK_SET);
            if(i == readFileTimes - 1)
                fread(_arrayInMem, sizeof(int), FileDigitNum - digitNum*(readFileTimes - 2), _fin);
            else
                fread(_arrayInMem, sizeof(int), digitNum, _fin);
        }
        if(_rank == 0){
//            printArr(_arrayInMem, digitNum, "arrayInMem:");
            PRINT("digitNum = %d\n", digitNum);
            PRINT("procDigitNum = %d\n", procDigitNum);
        }
        // 分发数据, Map
        MPI_Scatter(_arrayInMem, procDigitNum, MPI_INT,
                    procDigit, procDigitNum, MPI_INT, 0, MPI_COMM_WORLD);
//        if(_rank == 0)
//            printArr(procDigit, procDigitNum, "Process 0 Digit:");
        // 各个进程执行
        clock_t count_start = clock();
        countArray(procDigit, procDigitNum, num2cnt);
        clock_t count_End = clock();
        count_duration += (double)(count_End - count_start) / CLOCKS_PER_SEC;
//        if(_rank == 0)
//            printMap(num2cnt, "Process 0 Map:");
//        if(_rank == 1)
//            printMap(num2cnt, "Process 1 Map:");
//        if(_rank == 2)
//            printMap(num2cnt, "Process 2 Map:");
//        if(_rank == 3)
//            printMap(num2cnt, "Process 3 Map:");
//        if(_rank == 0)
//            printMap(digit2num, "Result Map:");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // 收集结果, Reduce
    clock_t reduce_start = clock();
    reduceResult(num2cnt, digit2num);
    clock_t reduce_End = clock();
    double reduce_duration = (double)(reduce_End - reduce_start) / CLOCKS_PER_SEC;
    if(_rank == 0){
        printf("Count duration = %.1lf\n Reduce duration = %.1f\n", count_duration, reduce_duration);
    }

    if(_rank == 0)
        delete []_arrayInMem;
    delete []procDigit;
}


void DigitCount::distributeTaskWithArray(short *masterArr) {

    MPI_File fh;
    MPI_Status status;
    MPI_File_open(MPI_COMM_WORLD, _inFile.c_str(),
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    int readFileTimes = ceil((double)_FileSize / _memoryCanBeUsed);
    // 所有进程开内存
    int *procDigit = new int[_memoryCanBeUsed / sizeof(int) / _size];
    if(_rank == 0)
            PRINT("readFileTimes = %d, _memoryCanBeUsed = %lld\n", readFileTimes, _memoryCanBeUsed);
    // 文件的int的数量
    long long FileDigitNum = _FileSize / sizeof(int);
    if(_rank == 0)
            PRINT("FileDigitNum = %lld\n", FileDigitNum);
    // 直接使用Array来进行计数
    short *cntArr = new short[ARRAY_SIZE];
    memset(cntArr, 0, sizeof(short)*ARRAY_SIZE);
    double count_duration = 0;
    double read_duration = 0;
    long long digitNum = (long long)(_memoryCanBeUsed / sizeof(int));
    for (int i = 0; i < readFileTimes; ++i) {
        // 每次主进程读取的digitNum
        digitNum = min(digitNum, FileDigitNum - digitNum*i);
        // 每个进程对应的数量
        long long procDigitNum = digitNum / _size;
        // 主进程读取数据
        clock_t read_start = clock();
        MPI_File_seek(fh, _rank*procDigitNum*sizeof(int) + _memoryCanBeUsed * i, MPI_SEEK_SET);
        MPI_File_read(fh, procDigit, procDigitNum, MPI_INT, &status);
        if(_rank == 0 && i == 0){
//            printArr(_arrayInMem, digitNum, "arrayInMem:");
            PRINT("digitNum = %lld\n", digitNum);
            PRINT("procDigitNum = %lld\n", procDigitNum);
        }
        clock_t read_End = clock();
        read_duration += (double)(read_End - read_start) / CLOCKS_PER_SEC;
//        if(_rank == 0)
//            printArr(procDigit, procDigitNum, "Process 0 Digit:");
        // 各个进程执行
//        map<int, long> num2cnt;
        clock_t count_start = clock();
        countArrayWithArray(procDigit, procDigitNum, cntArr);
        clock_t count_End = clock();
        count_duration += (double)(count_End - count_start) / CLOCKS_PER_SEC;

        MPI_Barrier(MPI_COMM_WORLD);
    }
    // 收集结果, Reduce
    clock_t reduce_start = clock();
    reduceResultWithArray(cntArr, masterArr);
    clock_t reduce_End = clock();
    double reduce_duration = (double)(reduce_End - reduce_start) / CLOCKS_PER_SEC;
    PRINT("RANK_%d: Read duration = %.1lf\t Count duration = %.1lf\t Reduce duration = %.1f\n",
            _rank, read_duration, count_duration, reduce_duration);

    if(_rank == 0)
        delete []cntArr;
    delete []procDigit;
}


/// 统计一个array的cnt, 可用openmp优化
/// \param arr
/// \param length
/// \param num2cnt
void DigitCount::countArray(const int *arr, long length, map<int, long> &num2cnt) {
//#pragma omp parallel shared(num2cnt, N, process_num) private(id)
    for (int i = 0; i < length; ++i) {
        if(num2cnt.count(arr[i]))
            ++num2cnt[arr[i]];
        else
            num2cnt[arr[i]] = 1;
    }
}

void DigitCount::countArrayWithArray(const int *arr, long long length, short *cntArr) {
    omp_set_num_threads(THREAD_NUM);
#pragma omp parallel for
    for (int i = 0; i < length; ++i) {
#pragma omp atomic
        ++cntArr[getIndex(arr[i])];
    }
}

/// 收集计算所有结果
/// \param num2cnt
/// \param digit2num
void DigitCount::reduceResult(map<int, long> &num2cnt, map<int, long> &digit2num) {

    MPI_Status stat;

    if(_rank == 0){
        int numSize = num2cnt.size();
        // 统计自身结果
        for (map<int, long>::iterator iter = num2cnt.begin(); iter != num2cnt.end(); ++iter) {
            if(digit2num.count(iter->first))
                digit2num[iter->first] += iter->second;
            else
                digit2num[iter->first] = iter->second;
        }
        // 统计其他进程的结果
        for (int source = 1; source < _size; ++source) {
            MPI_Recv(&numSize, 1, MPI_INT, source, source+100, MPI_COMM_WORLD, &stat);
            NumCnt *numcnt = new NumCnt[numSize];
            MPI_Recv(numcnt, numSize, MPI_NumCnt, source, source+1000, MPI_COMM_WORLD, &stat);

            for (int i = 0; i < numSize; ++i) {
                if(digit2num.count(numcnt[i].num))
                    digit2num[numcnt[i].num] += numcnt[i].cnt;
                else
                    digit2num[numcnt[i].num] = numcnt[i].cnt;
            }
            delete []numcnt;
        }
    }
    else{
        // 发送结果
        int numSize = num2cnt.size();
        MPI_Send(&numSize, 1, MPI_INT, 0, _rank+100, MPI_COMM_WORLD);
        NumCnt *numcnt = new NumCnt[numSize];
        int i = 0;
        // 可优化
        for (map<int, long>::iterator iter = num2cnt.begin();
                 iter != num2cnt.end() ; ++iter) {
            numcnt[i].num = iter->first;
            numcnt[i].cnt = iter->second;
            ++i;
        }
        if(numSize){
            MPI_Send(numcnt, numSize, MPI_NumCnt, 0, _rank+1000, MPI_COMM_WORLD);
        }
        delete []numcnt;
    }

}

void DigitCount::reduceResultWithArray(short *branchArr, short *masterArr) {
    MPI_Status stat;
    MPI_Reduce(branchArr, masterArr, ARRAY_SIZE>>1, MPI_SHORT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(branchArr + (ARRAY_SIZE>>1), masterArr + (ARRAY_SIZE>>1), ARRAY_SIZE>>1+1, MPI_SHORT, MPI_SUM, 0, MPI_COMM_WORLD);
}


void DigitCount::Count(map<int, long> &digit2num)
{
    clock_t start = clock();
    distributeTask(digit2num);
    clock_t End = clock();
    double duration = (double)(End - start) / CLOCKS_PER_SEC;
    // 因为Map还需要占用内存,所以总的内存使用量大概是1.2*Memory左右
    printf("Rank_%d: Use Memory %.1lf KB, Cost %f seconds\n", _rank, 1.2*MEMORY/1024, duration);
//    if(_rank == 0){
//        printMap(digit2num, "Result Map:");
//    }TEST_printResult(result);


    MPI_Finalize();
}

void DigitCount::CountWithArray(short *resultArr) {
    clock_t start = clock();
    distributeTaskWithArray(resultArr);
    clock_t End = clock();
    double duration = (double)(End - start) / CLOCKS_PER_SEC;
    double *durationArr = new double[_size];
    MPI_Gather(&duration, 1, MPI_DOUBLE, durationArr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // 因为Map还需要占用内存,所以总的内存使用量大概是1.2*Memory左右
    if(_rank == 0){
        double avg_duration = 0, max_duration = -1, min_duration = 9999999;
        for (int i = 0; i < _size; ++i) {
            avg_duration += durationArr[i];
            if(durationArr[i] > max_duration)   max_duration = durationArr[i];
            if(durationArr[i] < min_duration)   min_duration = durationArr[i];
        }
        printf("Total Use Memory %.1lf KB, Average Cost %.2lf seconds, Max Duration is %.2lf seconds, "
                       "Min Duration is %.2lf seconds\n"
                , (double)(ARRAY_SIZE*_size*sizeof(int) + MEMORY)/1024
                , avg_duration/_size, max_duration, min_duration);
    }
    MPI_Finalize();
//    if(_rank == 0){
//        printMap(digit2num, "Result Map:");
//    }TEST_printResult(result);
    //
}

void countDigit(char *filename){
    DigitCount counter(filename, MEMORY);
//    map<int, long>* digit2num = new map<int, long>();
//    counter.Count(*digit2num);
    //int *resultArr = new int[ARRAY_SIZE];
    short *resultArr = new short[ARRAY_SIZE];
    counter.CountWithArray(resultArr);
    delete []resultArr;
    //MPI_Finalize();
//    delete digit2num;
}


void TEST_calcAns(char *filename){
    FILE *_fin = fopen(filename, "r");


    struct stat fileStatus;
    stat(filename, &fileStatus);
    int _FileSize = fileStatus.st_size;
    int length = _FileSize / sizeof(int);
    int *arr = new int[length];

    fread(arr, sizeof(int), length, _fin);
    map<int, long> m;
    for (int i = 0; i < length; ++i) {
        if(m.count(arr[i]))
            ++m[arr[i]];
        else
            m[arr[i]] = 1;
    }
    printMap(m, "Ans:");
    delete []arr;
}

/// 运行方式: mpicxx digitcount.cpp -o digitcount -lpthread -std=c++11 -fopenmp -D__DEBUG__ && mpirun -np 2 ./digitcount filename (test)
/// mpicxx digitcount.cpp -o digitcount -lpthread -std=c++11 -fopenmp -D__DEBUG__ && yhrun -n 16 -p free --time=5 ./digitcount ../data128M.txt
int main(int argc, char* argv[]){
    std::cout<<ARRAY_SIZE<<endl;
    printf("int: %d\n", ARRAY_SIZE>>1);
    if(argc < 2){
        printf("Argument Error: Please Input file name\n");
    }
    if(argc == 3 && strcmp(argv[2], "test\n")){
        printf("In TEST Mode now, it will cost less MEMORY(ONLY USE WHEN value span is 0 - 1M.\n");
        ARRAY_SIZE = 1024*1024;
    }
    else printf("In NORMAL Mode now, it will cost more MEMORY\n");
//    TEST_calcAns(argv[1]);
    countDigit(argv[1]);
//    TEST_MAPFUNC();
}
