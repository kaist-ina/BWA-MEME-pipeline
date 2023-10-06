#include "bandedSWA.h"

namespace kernel{
    void test(void);
    void gpu_kernel_test(int *input, int *output);
    int get_gpu_num(void);
    void gpu_kernel_wrapper(SeqPair *pair_ar, uint8_t *seqBufQer, uint8_t *seqBufRef,
                int refSize, int qerSize, int numPair, int minSize, int maxRef, int maxQer,
               int8_t matchScore, int8_t misMatchScore, int8_t startGap, int8_t extendGap, int tid);
    void device_initialize(SeqPair *pair_ar_d, uint8_t *seqBufQer_d, uint8_t *seqBufRef_d,
                                int max_batch_size, int maxRef, int maxQer, int tid, int8_t *mat_d,
                                int8_t matchScore, int8_t misMatchScore);
    void device_free(SeqPair *pair_ar_d, uint8_t *seqBufQer_d, uint8_t *seqBufRef_d, int8_t *mat_d);

}

