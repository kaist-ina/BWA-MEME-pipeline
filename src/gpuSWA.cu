#include "gpuSWA.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string.h>
#include <thread>
#include <vector>

using namespace std;

#define cudaErrchk(ans)                                                                  \
{                                                                                    \
    gpuAssert((ans), __FILE__, __LINE__);                                            \
}
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){
    if(code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

void kernel::test(void) {
    printf("hello world\n");
    return;
}

__global__ void kernel_code(int *input, int *output, int length) {
    int thread_id = threadIdx.x;
    *(output + thread_id) = *(input + thread_id) + 2;
    // printf("threadId: %d, output: %d\n", thread_id, *(output+thread_id));
    return;
}

void kernel::gpu_kernel_test(int *input, int *output) {
    int *input_d, *output_d;
    cudaMalloc(&input_d, sizeof(int) * 5);
    cudaMalloc(&output_d, sizeof(int) * 5);
    cudaMemcpy(input_d, input, sizeof(int)*5, cudaMemcpyHostToDevice);
    fprintf(stderr, "input: ");
    for (int k = 0; k < 5; k++) {
        fprintf(stderr, "%d", *(input+k));
    }
    kernel_code<<<1, 5>>>(input_d, output_d, 5);
    
    fprintf(stderr, "\noutput: ");
    cudaMemcpy(output, output_d, sizeof(int)*5, cudaMemcpyDeviceToHost);
    for (int k = 0; k < 5; k++) {
        fprintf(stderr, "%d", *(output + k));
    }
    fprintf(stderr, "\n");
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceReset();
}

__device__ short findMaxFour(short a, short b, short c, short d) {
    short res = a;
    res = (res > b) ? res : b;
    res = (res > c) ? res : c;
    res = (res > d) ? res : d;
    return res;
}

__inline__ __device__ short warpReduceMax_with_index(short val, short &myIndex, short &myIndex2, unsigned lengthSeqB) {
    short myMax = val, newInd = 0, newInd2 = 0, ind = myIndex, ind2 = myIndex2;
    unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < lengthSeqB);
    int tempVal;
    
    for (int offset = 16; offset > 0; offset /= 2) {
        tempVal = __shfl_down_sync(mask, val, offset);
        newInd = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);
        val = max(val, tempVal);
        if (val != myMax) {
            ind = newInd;
            ind2 = newInd2;
            myMax = val;
        }
        else if (val == tempVal) {
            if (newInd < ind || ((newInd == ind) && (newInd2 > ind2))) {
                ind = newInd;
                ind2 = newInd2;
            }
        }
    }
    val = myMax;
    myIndex = ind;
    myIndex2 = ind2;
    return val;
    // printf("%d, %d\n", myIndex, myIndex2);
    // printf("");
}

__device__ short blockShuffleReduce_with_index(short myVal, short &myIndex, short &myIndex2, unsigned lengthSeqB) {
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short myInd = myIndex;
    short myInd2 = myIndex2;
    myVal = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB);
    
    __syncthreads();

    if (laneId == 0) {
        locTots[warpId] = myVal;
        locInds[warpId] = myInd;
        locInds2[warpId] = myInd2;
    }
    __syncthreads();

    unsigned check = ((32 + blockDim.x - 1) / 32);
    if (threadIdx.x < check) {
        myVal = locTots[threadIdx.x];
        myInd = locInds[threadIdx.x];
        myInd2 = locInds2[threadIdx.x];
    }
    else {
        myVal = 0;
        myInd = -1;
        myInd2 = -1;
    }
    __syncthreads();

    if (warpId == 0) {
        myVal = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB);
        myIndex = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
    return myVal;
}

__global__ void bsw_kernel(uint8_t *seqA_array, uint8_t *seqB_array, SeqPair *pair_ar, int8_t matchScore, int8_t misMatchScore,
                            int8_t startGap, int8_t extendGap, int8_t *mat)
{
    
    int block_Id = blockIdx.x;
    int thread_Id = threadIdx.x;
    short laneId = threadIdx.x % 32;
    short laneId2 = thread_Id % 32;
    
    short warpId = threadIdx.x/32;
    unsigned lengthSeqA, lengthSeqB;

    // local pointer
    uint8_t *seqA, *seqB;
    extern __shared__ char is_valid_array[];
    char *is_valid = &is_valid_array[0];
    uint8_t *longer_seq;

    // setting up block local sequences and their lengths
    SeqPair *pair = pair_ar + block_Id;
    lengthSeqA = pair->len1;
    lengthSeqB = pair->len2;
    /*------------------------------------------------*/
    if (lengthSeqA == 0 || lengthSeqB == 0) {
        pair->score = pair->h0;
        pair->gscore = -1;
        return;
    }
    // pair->score = pair->h0;
    // pair->gscore = -1;
    /*------------------------------------------------*/
    seqA = seqA_array + pair->idr;
    seqB = seqB_array + pair->idq;
    
    int32_t maxSize, minSize;
    /*------------------------------------------------*/
    // maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    // minSize = lengthSeqA <= lengthSeqB ? lengthSeqA : lengthSeqB;
    maxSize = lengthSeqA;
    minSize = lengthSeqB;
    /*------------------------------------------------*/
    // for (int p = thread_Id; p < minSize; p+=32) {
    //     is_valid[p] = 0;
    // }
    // is_valid += minSize;
    // for (int p = thread_Id; p < maxSize; p+=32) {
    //     is_valid[p] = 1;
    // }
    // is_valid += maxSize;
    // for (int p = thread_Id; p < minSize; p+=32) {
    //     is_valid[p] = 0;
    // }
    for (int p = thread_Id; p < minSize; p+=32) {
        is_valid[p] = 0;
    }
    is_valid += minSize;
    for (int p = thread_Id; p < minSize; p+=32) {
        is_valid[p] = 1;
    }
    is_valid += minSize;
    for (int p = thread_Id; p < minSize; p+=32) {
        is_valid[p] = 0;
    }
    
    uint8_t myColumnChar;
    
    /*------------------------------------------------*/
    // if (lengthSeqA <= lengthSeqB) {
    //     if (thread_Id < lengthSeqA) myColumnChar = seqA[thread_Id];
    //     longer_seq = seqB;
    // }
    // else {
    //     if (thread_Id < lengthSeqB) myColumnChar = seqB[thread_Id];
    //     longer_seq = seqA;
    // }
    if (thread_Id < lengthSeqB) myColumnChar = seqB[thread_Id];
    longer_seq = seqA;
    /*------------------------------------------------*/
    __syncthreads();

    // initialize registers
    int i;
    short thread_max, thread_max_i, thread_max_j;
    i = 1; thread_max = thread_max_i = thread_max_j = 0;
    short thread_gmax, thread_max_gi, thread_max_gj;
    thread_gmax = thread_max_gi = thread_max_gj = 0;

    short _curr_H = 0, _curr_F = 0, _curr_E = 0;
    short _prev_H = 0, _prev_F = 0, _prev_E = 0;
    short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
    short _temp_Val = 0;
    short _curr_D = 0, _prev_D = 0;

    __shared__ short sh_prev_E[32];
    __shared__ short sh_prev_H[32];
    __shared__ short sh_prev_prev_H[32];

    __shared__ short local_spill_prev_E[1024];
    __shared__ short local_spill_prev_H[1024];
    __shared__ short local_spill_prev_prev_H[1024];

    __syncthreads();
    
    for (int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++) {
        // is_valid = is_valid - 1;
        is_valid = is_valid - (diag < minSize || diag >= maxSize);

        _temp_Val = _prev_H;
        _prev_H = _curr_H;
        _curr_H = _prev_prev_H;
        _prev_prev_H = _temp_Val;
        _curr_H = 0;
        
        _prev_D = _curr_D;
        _curr_D = 0;

        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E = 0;

        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
        _curr_F = 0;

        if (laneId == 31) { // if you are the last thread in your warp then spill your values to shmem
            sh_prev_E[warpId] = _prev_E;
            sh_prev_H[warpId] = _prev_D;
            // sh_prev_H[warpId] = _prev_H;
            sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if (diag >= maxSize) { // if you are invalid in this iteration, spill your values to shmem
            local_spill_prev_E[thread_Id] = _prev_E;
            local_spill_prev_H[thread_Id] = _prev_D;
            // local_spill_prev_H[thread_Id] = _prev_H;
            local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
        }
        __syncthreads();
        
        if (is_valid[thread_Id] && thread_Id < minSize) {
            unsigned mask = __ballot_sync (__activemask(), (is_valid[thread_Id] && (thread_Id < minSize)));

            short fVal = _prev_F + extendGap;
            short hfVal = _prev_D + startGap;
            // short hfVal = _prev_H + startGap;
            short valeShfl = __shfl_sync(mask, _prev_E, laneId-1, 32);
            short valheShfl = __shfl_sync(mask, _prev_D, laneId-1, 32);
            // short valheShfl = __shfl_sync(mask, _prev_H, laneId-1, 32);
            short eVal = 0, heVal = 0;

            // when the previous thread has phased out, get value from shmem
            if (diag >= maxSize) {
                eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
                heVal = local_spill_prev_H[thread_Id - 1] + startGap;
            }
            else {
                eVal = ((warpId != 0 && laneId == 0) ? sh_prev_E[warpId-1] : valeShfl) + extendGap;
                heVal = ((warpId != 0 && laneId == 0) ? sh_prev_H[warpId-1] : valheShfl) + startGap;
            }
            // make sure that values for lane 0 in warp 0 is not undefined
            if (warpId == 0 && laneId == 0) {
                eVal = 0;
                heVal = 0;
            }

            _curr_F = (fVal > hfVal) ? fVal : hfVal;
            _curr_F = (_curr_F > 0) ? _curr_F : 0;
            _curr_E = (eVal > heVal) ? eVal : heVal;
            _curr_E = (_curr_E > 0) ? _curr_E : 0;
            short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
            short final_prev_prev_H = 0;

            if (diag >= maxSize) {
                final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
            }
            else {
                final_prev_prev_H = (warpId != 0 && laneId == 0) ? sh_prev_prev_H[warpId - 1] : testShufll;
            }

            // if (i == 1) final_prev_prev_H = pair->h0 + (thread_Id * startGap);
            if (i == 1) final_prev_prev_H = pair->h0 + min(1, thread_Id) * startGap + max(0, thread_Id-1)*extendGap;
            else if (warpId == 0 && laneId == 0) {
                final_prev_prev_H = pair->h0 + (startGap - extendGap) + extendGap*(i - 1);
            }
            uint8_t to_comp = longer_seq[i - 1];

            // using scoring matrix
            // short comp_score = mat[to_comp*5 + myColumnChar];
            short comp_score = (to_comp == myColumnChar) ? matchScore : misMatchScore;
            comp_score = (to_comp == 4 || myColumnChar == 4) ? -1 : comp_score;
            // short diag_score = final_prev_prev_H ? final_prev_prev_H + mat[to_comp*5 + myColumnChar] : 0;
            short diag_score = final_prev_prev_H ? final_prev_prev_H + comp_score : 0;
            _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0);
            _curr_D = max(diag_score, 0);

            int _i = i;
            int _j = thread_Id + 1;
            thread_max_i = (thread_max < _curr_H) ? _i : thread_max_i;
            thread_max_j = (thread_max < _curr_H) ? _j : thread_max_j;
            thread_max = (thread_max < _curr_H) ? _curr_H : thread_max;
            
            /*----------------------------------------------*/
            thread_max_gi = (thread_gmax <= _curr_H) ? _i : thread_max_gi;
            thread_max_gj = (thread_gmax <= _curr_H) ? _j : thread_max_gj;
            thread_gmax = (thread_gmax <= _curr_H) ? _curr_H : thread_gmax;
            /*----------------------------------------------*/
            i++;
        } // if is_valid end
        __syncthreads();
    } // for diag end
    __syncthreads();
    

    // find thread max;
    // if (thread_Id != 0) thread_max += 2;
    /*----------------------------------------------------------------*/
    if (thread_Id == minSize-1) {
        // if (lengthSeqA <= lengthSeqB) {
        //     pair->gscore = thread_gmax;
        //     pair->gtle = thread_max_gj;
        // }
        // else {
        //     pair->gscore = thread_gmax;
        //     pair->gtle = thread_max_gi;
        // }
        pair->gscore = thread_gmax;
        pair->gtle = thread_max_gi;
    }
    /*----------------------------------------------------------------*/
    thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j, minSize);
    // Assign value
    /*------------------------------------------------------------*/
    // if (thread_Id == 0) {
    //     pair->score = thread_max;
    //     pair->qle = thread_max_j;
    //     pair->tle = thread_max_i;
    //     // pair->gscore = thread_max;
    //     // pair->gtle = thread_max_i;
    //     pair->max_off = 0;
    // }
    if (thread_Id == 0) {
        if (thread_max <= pair->h0) {
            pair->score = pair->h0;
            pair->qle = 0;
            pair->tle = 0;
        }
        // else if (lengthSeqA <= lengthSeqB) {
        //     pair->score = thread_max;
        //     pair->qle = thread_max_i;
        //     pair->tle = thread_max_j;
        //     // pair->gscore = thread_max;
        //     // pair->gtle = thread_max_j;
        //     pair->max_off = 0;
        // }
        else {
            pair->score = thread_max;
            pair->qle = thread_max_j;
            pair->tle = thread_max_i;
            // pair->gscore = thread_max;
            // pair->gtle = thread_max_i;
            pair->max_off = 0;
        }
        // else if (lengthSeqA > lengthSeqB) {
        //     pair->score = thread_max;
        //     pair->qle = thread_max_j;
        //     pair->tle = thread_max_i;
        //     // pair->gscore = thread_max;
        //     // pair->gtle = thread_max_i;
        //     pair->max_off = 0;
        // }
    }
    /*------------------------------------------------------------*/
    return;
}

int kernel::get_gpu_num(void) {
    int gpu_num;
    cudaGetDeviceCount(&gpu_num);
    fprintf(stderr, "gpu: %d\n", gpu_num);
    return gpu_num;
}
// void kernel::device_initialize(SeqPair *pair_ar_d, uint8_t *seqBufQer_d, uint8_t *seqBufRef_d,
//                                 int max_batch_size, int maxRef, int maxQer, int tid, int8_t *mat_d,
//                                 int8_t matchScore, int8_t misMatchScore) {
//     cudaErrchk(cudaSetDevice(tid%2));
//     int8_t *mat;
//     mat = (int8_t *) calloc(25, sizeof(int8_t));
//     int p, q, k;
//     for (p = k = 0; p < 4; ++p) {
//         for (q = 0; q < 4; ++q) {
//             mat[k++] = p == q? matchScore : -1 * misMatchScore;
//         }
//         mat[k++] = -1;
//     }
//     for (q = 0; q < 5; ++q) mat[k++] = -1;
//     cudaMalloc(&mat_d, sizeof(int8_t) * 25);
//     cudaMemcpy(mat_d, mat, sizeof(int8_t) * 25, cudaMemcpyHostToDevice);
    
//     cudaErrchk(cudaMalloc(&seqBufRef_d, sizeof(uint8_t) * maxRef * max_batch_size));
//     cudaErrchk(cudaMalloc(&seqBufQer_d, sizeof(uint8_t) * maxQer * max_batch_size));
//     cudaErrchk(cudaMalloc(&pair_ar_d, sizeof(SeqPair) * max_batch_size));
//     free(mat);
//     fprintf(stderr, "mem\n");
//     return;
// }
// void kernel::device_free(SeqPair *pair_ar_d, uint8_t *seqBufQer_d, uint8_t *seqBufRef_d, int8_t *mat_d)
// {
//     cudaFree(seqBufRef_d);
//     cudaFree(seqBufQer_d);
//     cudaFree(pair_ar_d);
//     cudaFree(mat_d);
// }

void kernel::gpu_kernel_wrapper(SeqPair *pair_ar, uint8_t *seqBufQer, uint8_t *seqBufRef,
    int refSize, int qerSize, int numPair, int minSize, int maxRef, int maxQer,
    int8_t matchScore, int8_t misMatchScore, int8_t startGap, int8_t extendGap, int tid)
{
    SeqPair *pair_ar_d;
    uint8_t *seqBufRef_d;
    uint8_t *seqBufQer_d;
    int max_batch_size = numPair;
    // fprintf(stderr, "numPair: %d, tid: %d, refSize: %d, qerSize: %d, maxRef: %d, maxQer: %d\n", numPair, tid, refSize, qerSize, maxRef, maxQer);
    cudaErrchk(cudaSetDevice(tid%4));

    // // scoring matrix optimization
    int8_t *mat, *mat_d;
    // mat = (int8_t *) calloc(25, sizeof(int8_t));
    // int p, q, k;
    // for (p = k = 0; p < 4; ++p) {
    //     for (q = 0; q < 4; ++q) {
    //         mat[k++] = p == q? matchScore : -1 * misMatchScore;
    //     }
    //     mat[k++] = -1;
    // }
    // for (q = 0; q < 5; ++q) mat[k++] = -1;
    // cudaMalloc(&mat_d, sizeof(int8_t) * 25);
    // cudaMemcpy(mat_d, mat, sizeof(int8_t) * 25, cudaMemcpyHostToDevice);
    // int gpu_num;
    // cudaGetDeviceCount(&gpu_num);
    // fprintf(stderr, "gpu: %d\n", gpu_num);
    // create stream
    // cudaStream_t my_stream;
    // cudaEvent_t data_event;
    // cudaStreamCreate(&my_stream);
    // cudaEventCreateWithFlags(&data_event, cudaEventBlockingSync);

    // fprintf(stderr, "thread %d aa\n", tid);
    // Get batch size
    int left_size, iter = 0;
    if (numPair <= max_batch_size) {
        left_size = numPair;
    }
    else {
        left_size = numPair;
        // fprintf(stderr, "leftsize: %d\n", left_size);
        while (left_size > max_batch_size) {
            iter++;
            left_size -= max_batch_size;
            // fprintf(stderr, "left_size: %d\n", left_size);
        }
    }
    // fprintf(stderr, "numpair: %d, iter: %d, left_size: %d\n", numPair, iter, left_size);
    // Get shared memory size
    unsigned totShmem = 3 * (minSize + 1) * sizeof(int32_t);
    unsigned alignmentPad = 4 + (4 - totShmem % 4);
    size_t ShmemBytes = totShmem + alignmentPad;
    
    // host pinned memory allocation
    // uint8_t *seqBufRef_pin, *seqBufQer_pin;
    // SeqPair *pair_ar_pin;
    // cudaErrchk(cudaMallocHost(&seqBufRef_pin, sizeof(uint8_t) * maxRef * max_batch_size));
    // cudaErrchk(cudaMallocHost(&seqBufQer_pin, sizeof(uint8_t) * maxQer * max_batch_size));
    // cudaErrchk(cudaMallocHost(&pair_ar_pin, sizeof(SeqPair) * max_batch_size));

    // device memory allocation
    cudaErrchk(cudaMalloc(&seqBufRef_d, sizeof(uint8_t) * refSize));
    cudaErrchk(cudaMalloc(&seqBufQer_d, sizeof(uint8_t) * qerSize));
    cudaErrchk(cudaMalloc(&pair_ar_d, sizeof(SeqPair) * max_batch_size));

    // fprintf(stderr, "thread %d bb\n", tid);
    int batch_size, pair_off = 0, ref_off = 0, qer_off = 0, ref_len, qer_len;
    SeqPair *tmp_pair, *first, *last;
    uint8_t *tmpRef, *tmpQer;
    if(ShmemBytes > 48000)
        cudaFuncSetAttribute(bsw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

    // fprintf(stderr, "iter: %d batch: %d, ref_off: %d, qer_off: %d, pair_off: %d, ref_len: %d, qer_len: %d\n", i, batch_size, ref_off, qer_off, pair_off, ref_len, qer_len);
    batch_size = numPair;
    cudaErrchk(cudaMemcpy(pair_ar_d, pair_ar, sizeof(SeqPair)*batch_size, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(seqBufRef_d, seqBufRef, sizeof(uint8_t)*refSize, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(seqBufQer_d, seqBufQer, sizeof(uint8_t)*qerSize, cudaMemcpyHostToDevice));

    bsw_kernel<<<batch_size, minSize, ShmemBytes>>>(seqBufRef_d, seqBufQer_d, pair_ar_d, matchScore, -1 * misMatchScore, -1*startGap, -1*extendGap, mat_d);
    
    // cudaMemcpy(pair_ar, pair_ar_d, sizeof(SeqPair) * batch_size, cudaMemcpyDeviceToHost);
    cudaErrchk(cudaMemcpy(pair_ar, pair_ar_d, sizeof(SeqPair) * batch_size, cudaMemcpyDeviceToHost));
    
    // pair_off += batch_size;
    // ref_off += ref_len;
    // qer_off += qer_len;
    
    // for (int i = 0; i <= iter; i++) {
    //     if (i == iter) batch_size = left_size;
    //     else batch_size = max_batch_size;
        
    //     tmp_pair = pair_ar + pair_off;
    //     tmpRef = seqBufRef + ref_off;
    //     tmpQer = seqBufQer + qer_off;
        
    //     // Get length of query and reference
    //     first = pair_ar + pair_off;
    //     last = first + batch_size - 1;
    //     ref_len = last->idr + last->len1 - first->idr;
    //     qer_len = last->idq + last->len2 - first->idq;
    //     // fprintf(stderr, "iter: %d batch: %d, ref_off: %d, qer_off: %d, pair_off: %d, ref_len: %d, qer_len: %d\n", i, batch_size, ref_off, qer_off, pair_off, ref_len, qer_len);

    //     memcpy(seqBufRef_pin, tmpRef, sizeof(uint8_t)*ref_len);
    //     memcpy(seqBufQer_pin, tmpQer, sizeof(uint8_t)*qer_len);
    //     memcpy(pair_ar_pin, tmp_pair, sizeof(uint8_t)*batch_size);

    //     cudaErrchk(cudaMemcpyAsync(seqBufRef_d, seqBufRef_pin, sizeof(uint8_t)*ref_len, cudaMemcpyHostToDevice, my_stream));
    //     cudaErrchk(cudaMemcpyAsync(seqBufQer_d, seqBufQer_pin, sizeof(uint8_t)*qer_len, cudaMemcpyHostToDevice, my_stream));
    //     cudaErrchk(cudaMemcpyAsync(pair_ar_d, pair_ar_pin, sizeof(SeqPair)*batch_size, cudaMemcpyHostToDevice, my_stream));

    //     bsw_kernel<<<batch_size, minSize, ShmemBytes, my_stream>>>(seqBufRef_d, seqBufQer_d, pair_ar_d, matchScore, -1 * misMatchScore, -1*startGap, -1*extendGap, mat_d);
    //     cudaErrchk(cudaMemcpyAsync(pair_ar_pin, pair_ar_d, sizeof(SeqPair) * batch_size, cudaMemcpyDeviceToHost, my_stream));
        
    //     // cudaErrchk(cudaEventRecord(data_event, my_stream));
    //     // cudaErrchk(cudaEventSynchronize(data_event));

    //     memcpy(tmp_pair, pair_ar_pin, sizeof(SeqPair)*batch_size);
    //     // fprintf(stderr, "iter: %d kernel code done\n", i);
    //     pair_off += batch_size;
    //     ref_off += ref_len;
    //     qer_off += qer_len;
    //     cudaStreamSynchronize(my_stream);
    // }
    // fprintf(stderr, "thread %d cc\n", tid);

    // zero debugging
    // for (int i = 0; i < numPair; i++) {
    //     tmp_pair = pair_ar + i;
    //     if (tmp_pair->score == 0 || tmp_pair->score == -1) {
    //         fprintf(stderr, "pair %d, init: %d, score: %d, tle: %d, qle: %d, refLen: %d, qerLen: %d\n", i, tmp_pair->h0, tmp_pair->score, tmp_pair->tle, tmp_pair->qle, tmp_pair->len1, tmp_pair->len2);
    //         fprintf(stderr, "ori ref: ");
    //         for (int uu = 0; uu < tmp_pair->len1; uu++) {
    //             fprintf(stderr, "%u ", *(seqBufRef + tmp_pair->idr + uu));
    //         }
    //         fprintf(stderr, "\n");
    //         fprintf(stderr, "ori qer: ");
    //         for (int uu = 0; uu < tmp_pair->len2; uu++) {
    //             fprintf(stderr, "%u ", *(seqBufQer + tmp_pair->idq + uu));
    //         }
    //         fprintf(stderr, "\n");
    //     }
    // }

    // if (pair_off != numPair) fprintf(stderr, "off: %d, pair: %d\n", pair_off, numPair);
    // if (ref_off != refSize) fprintf(stderr, "off: %d, ref: %d\n", ref_off, refSize);
    // if (qer_off != qerSize) fprintf(stderr, "off: %d, qer: %d\n", qer_off, qerSize);

    cudaFree(seqBufRef_d);
    cudaFree(seqBufQer_d);
    cudaFree(pair_ar_d);
    // cudaFreeHost(seqBufRef_pin);
    // cudaFreeHost(seqBufQer_pin);
    // cudaFreeHost(pair_ar_pin);
    // free(mat);
    // cudaFree(mat_d);
    // cudaStreamDestroy(my_stream);
    // cudaEventDestroy(data_event);
    // fprintf(stderr, "done tid: %d\n", tid);./
    return;
}
