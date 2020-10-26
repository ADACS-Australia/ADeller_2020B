#include <iostream>
#include <cstdlib>
#include <cufft.h>

using namespace std;

#ifndef PROCESSOR_GPU
#define PROCESSOR_GPU

#define u8                  uint8_t
#define f32                 float
#define cf32                cufftComplex

#define checkCuda(err) __checkCuda(err, (char *)__FILE__, __LINE__)
inline cudaError_t __checkCuda(cudaError_t err, char *file, int line) {
    if (err != cudaSuccess) {
        cerr << "Error in calling CUDA operation in " << file << " at line " << line << endl;
    }
    return err;
}

typedef struct {
    int length;
    cf32 *in;
    cf32 *out;

    // TODO - worry about the following bits
    // int flag;
    // hint
    // int *wbufsize;
    // u8 **fftworkbuf;
} ProcessorContext;

class GpuModeProcessor {

    public:
        GpuModeProcessor();
        ~GpuModeProcessor();
        void process(ProcessorContext *context);

};

typedef struct {
    cufftHandle p;

    // device
    cf32 *d_in;
    cf32 *d_out;

    int len;
    int len2;
    int len3;
} cufftPtr_cf32;

// TODO - what is 'hint' need to add it.
#define cufftInitC2Cf32(fftspec, context) __cufftInitC2Cf32(fftspec, context)
inline int __cufftInitC2Cf32(cufftPtr_cf32 **fftspec, ProcessorContext *context) {
    // allocate cufft ptr
    fftspec[0] = (cufftPtr_cf32 *)malloc(sizeof(cufftPtr_cf32));

    // cufft plan - Nvidia says do this first else there can be memory allocation errors
    cufftPlan1d(&(fftspec[0]->p), context->length, CUFFT_C2C, 1);

    // device allocate
    checkCuda(cudaMalloc((void **) &(fftspec[0]->d_in), context->length*sizeof(cf32)) );
    checkCuda(cudaMalloc((void **) &(fftspec[0]->d_out), context->length*sizeof(cf32)) );

    // copy from host to device
    checkCuda( cudaMemcpy(fftspec[0]->d_in, context->in, context->length*sizeof(cf32), cudaMemcpyHostToDevice) );

    // lengths
    fftspec[0]->len = context->length;
    fftspec[0]->len2 = 1;
    fftspec[0]->len3 = 1;

    return 0;
}

#define cufftFreeC2Cf32(fftspec) __cufftFreeC2Cf32(fftspec)
inline int __cufftFreeC2Cf32(cufftPtr_cf32 *fftspec) {
    // need to free plan, host, device and
    checkCuda( cudaFree(fftspec->d_in));
    checkCuda( cudaFree(fftspec->d_out));

    cufftDestroy(fftspec->p);

    free(fftspec);
    return 0;
}

#define createContext(context_ptr, length) __createContext(context_ptr, length)
inline void __createContext(ProcessorContext **context_ptr, int length) {
    ProcessorContext *context = (ProcessorContext *)malloc(sizeof(ProcessorContext));
    context->length = length;
    context->in = (cf32 *)malloc(length * sizeof(cf32));
    context->out = (cf32 *)malloc(length * sizeof(cf32));

    context_ptr[0] = context;
}

#define freeContext(context) __freeContext(context)
inline void __freeContext(ProcessorContext *context) {
    free(context->in);
    free(context->out);

    free(context);
}

#define deviceToHostCopyCf32(device, context) __deviceToHostCopyCf32(device, context)
inline int __deviceToHostCopyCf32(cf32 *device, ProcessorContext *context) {
    return checkCuda( cudaMemcpy(context->out, device, context->length * sizeof(cf32), cudaMemcpyDeviceToHost));
}

#endif
