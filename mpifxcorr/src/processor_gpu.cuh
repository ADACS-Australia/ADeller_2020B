#ifndef PROCESSOR_GPU
#define PROCESSOR_GPU

#include <iostream>
#include <cstdlib>
#include <cufft.h>

using namespace std;

#define BLOCK_SIZE          512

#define processResult       int

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
    cufftHandle p;
} cufftPtr_cf32;

typedef struct {
    int length;

    cf32 *complexrotator;
    cf32 *curr_unpacked;
    cf32 *fftd;
} ProcessorContext;

class GpuModeProcessor {

    public:
        GpuModeProcessor(int fftchannels);
        ~GpuModeProcessor();

        processResult process(ProcessorContext *context);

    protected:
        processResult perform_fft(ProcessorContext *context);
        processResult handle_sideband(ProcessorContext *context);

    private:
        int fftchannels;
        cufftPtr_cf32 *fftspec;

        // device
        cf32 *d_complexrotator;
        cf32 *d_curr_unpacked;
        cf32 *d_fftd;
};

#define createContext(context_ptr, length) __createContext(context_ptr, length)
inline int __createContext(ProcessorContext **context_ptr, int length) {
    ProcessorContext *context = (ProcessorContext *)malloc(sizeof(ProcessorContext));
    context->length = length;

    context->complexrotator = (cf32 *)malloc(length * sizeof(cf32));
    context->curr_unpacked = (cf32 *)malloc(length * sizeof(cf32));
    context->fftd = (cf32 *)malloc(length * sizeof(cf32));

    context_ptr[0] = context;

    return 0;
}

#define freeContext(context) __freeContext(context)
inline int __freeContext(ProcessorContext *context) {
    free(context->complexrotator);
    free(context->curr_unpacked);
    free(context->fftd);

    free(context);

    return 0;
}

// #define deviceToHostCopyCf32(device, context) __deviceToHostCopyCf32(device, context)
// inline int __deviceToHostCopyCf32(cf32 *device, ProcessorContext *context) {
//     return checkCuda( cudaMemcpy(context->out, device, context->length * sizeof(cf32), cudaMemcpyDeviceToHost));
// }

// #define hostToDeviceCopyCf32(dest, source) __deviceToHostCopyCf32(dest, source)
// inline int __deviceToHostCopyCf32(cf32 *dest, ProcessorContext *source) {
//     return checkCuda( cudaMemcpy(context->out, device, context->length * sizeof(cf32), cudaMemcpyDeviceToHost));
// }


#endif
