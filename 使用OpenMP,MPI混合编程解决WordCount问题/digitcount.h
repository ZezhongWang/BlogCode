//
// Created by w2w on 18-7-5.
//

#ifndef EXTERNALSORT_DIGITCOUNT_H_H
#define EXTERNALSORT_DIGITCOUNT_H_H

#include <iostream>
#include <string.h>
#include <map>
#include "mpi.h"

using std::string;
using std::map;

typedef struct{
    long cnt;
    int num;
}NumCnt;

class DigitCount{
public:
    DigitCount(string inFile, long memoryCanBeUsed);
    void Count(map<int, long>& digit2num);
    void CountWithArray(short *resultArr);
private:

    void defineTransportStruct();
    void distributeTask(map<int, long> &digit2num);
    void distributeTaskWithArray(short *masterArr);
    void countArray(const int *arr, long length, map<int, long> &num2cnt);
    void countArrayWithArray(const int*arr, long long length, short* cntArr);
    void reduceResult(map<int, long> &num2cnt, map<int, long>& digit2num);
    void reduceResultWithArray(short* branchArr, short* masterArr);
    MPI_Datatype MPI_NumCnt;
    string _inFile;
    FILE *_fin; // only proc 0
    int _rank;
    int _size;
    long long _FileSize; // only proc 0
    long long _memoryCanBeUsed;
    int *_arrayInMem; // only proc 0
};


#endif //EXTERNALSORT_DIGITCOUNT_H_H
