#include "processor_gpu.cuh"
#include <iostream>
#include <math.h>
using namespace std;

__global__ void cudaMul_cf32(cf32 *src1, cf32 *src2, cf32 *dest, int length) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += stride) {
        dest[i].x = src1[i].x * src2[i].x - src1[i].y * src2[i].y;
        dest[i].y = src1[i].x * src2[i].y + src1[i].y * src2[i].x;
    }
}

GpuModeProcessor::GpuModeProcessor(int fftchannels): fftchannels(fftchannels) {
    // init fftspec
    fftspec = (cufftPtr_cf32 *)malloc(sizeof(cufftPtr_cf32));

    cufftPlan1d(&(fftspec->p), fftchannels, CUFFT_C2C, 1);
    checkCuda(cudaMalloc((void **) &d_complexrotator, fftchannels*sizeof(cf32)) );
    checkCuda(cudaMalloc((void **) &d_curr_unpacked, fftchannels*sizeof(cf32)) );
    checkCuda(cudaMalloc((void **) &d_fftd, fftchannels*sizeof(cf32)) );
}

GpuModeProcessor::~GpuModeProcessor() {
    // need to free device mem
    checkCuda( cudaFree(d_complexrotator));
    checkCuda( cudaFree(d_curr_unpacked));
    checkCuda( cudaFree(d_fftd));

    // release the cuff plan
    cufftDestroy(fftspec->p);

    free(fftspec);
}

processResult GpuModeProcessor::process(ProcessorContext *context) {
    // copy from host to device
    checkCuda( cudaMemcpy(d_complexrotator, context->complexrotator, context->length*sizeof(cf32), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_curr_unpacked, context->curr_unpacked, context->length*sizeof(cf32), cudaMemcpyHostToDevice) );

    // perform the complex multiply
    dim3 dimGrid((context->length) / BLOCK_SIZE + 1);
    dim3 dimBlock(BLOCK_SIZE);
    cudaMul_cf32<<<dimGrid, dimBlock>>>(d_complexrotator, d_curr_unpacked, d_fftd, context->length);

    perform_fft(context);

    // copy device back to host
    checkCuda( cudaMemcpy(context->fftd, d_fftd, context->length * sizeof(cf32), cudaMemcpyDeviceToHost));

    return 0;
}

processResult GpuModeProcessor::perform_fft(ProcessorContext *context) {
    // perform the actual fft
    cufftExecC2C(fftspec->p, d_fftd, d_fftd, CUFFT_FORWARD);

    return 0;
}

processResult GpuModeProcessor::handle_sidebands(ProcessorContext *context) {
    // copy from fftd to the fftouputs[j][subloopindex]
    // vectorCopy_cf32(fftd, fftoutputs[j][subloopindex], recordedbandchannels);
}

#define TWO_PI 6.2831853072f
#define DEG_TO_RAD 0.03490658504f
#define FFT_LENGTH 512

int main() {
    GpuModeProcessor *processor = new GpuModeProcessor(FFT_LENGTH);

    ProcessorContext *context;
    createContext(&context, FFT_LENGTH);

    // prepare data
    for (auto i = 0; i < 512; i++) {
        context->complexrotator[i].x = cos(TWO_PI * i / 512);
        context->complexrotator[i].y = sin(2./3. * TWO_PI * i / 512 + 45 * DEG_TO_RAD);
        // it's just 1, the multiply should work
        context->curr_unpacked[i].x = 1;
        context->curr_unpacked[i].y = 0;
    }

    cout << "Context length = " << context->length << endl;

    processor->process(context);

    // get the data out
    for (auto i = 0; i < 512; i++) {
        cout << "fftd[" << i << "]: (" << context->fftd[i].x << ", " << context->fftd[i].y << ")" << endl;
    }

    freeContext(context);
    free(processor);
}
