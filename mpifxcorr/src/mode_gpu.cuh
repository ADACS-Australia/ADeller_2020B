#ifndef MODE_GPU_H
#define MODE_GPU_H

#include "cufft.h"

#define BLOCK_SIZE  512
#define cf32        cufftComplex

int create_cufftPlan(cufftHandle **plan, int length);
int free_cufftPlan(cufftHandle *plan);

int process(cf32 *complexrotator, cf32 *unpackedcomplex, cf32 *fftd, cf32 *fracsamprotator, int length, cufftHandle plan);

#endif
