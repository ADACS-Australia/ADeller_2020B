#include "processor_gpu.h"
#include <iostream>
#include <math.h>
using namespace std;

GpuModeProcessor::GpuModeProcessor() {
}

GpuModeProcessor::~GpuModeProcessor() {
}

void GpuModeProcessor::process(ProcessorContext *context) {
    // need to get length, flat, hint, buffersize, and buffer from context
    cufftPtr_cf32 *pFFTSpecC;

    // just do the FFT
    int status;
    status = cufftInitC2Cf32(&pFFTSpecC, context);

    if (status != 0) {
        cerr << "Error in FFT initialisation!!!" << status << endl;
    }

    // Do the FFT
    status = cufftExecC2C(pFFTSpecC->p, (cufftComplex *)pFFTSpecC->d_in, (cufftComplex *)pFFTSpecC->d_out, CUFFT_FORWARD);
    if (status != 0) {
        cerr << "Error in perform FFT spec" << status << endl;
    }

    // need to move data back from device to host
    status = deviceToHostCopyCf32(pFFTSpecC->d_out, context);
    if (status != 0) {
        cerr << "Error in copying from device to host" << status << endl;
    }

    // need to release the spec
    status = cufftFreeC2Cf32(pFFTSpecC);
    if (status != 0) {
        cerr << "Error in freeing FFT spec" << status << endl;
    }
}

#define TWO_PI 6.2831853072f
#define DEG_TO_RAD 0.03490658504f

int main() {
    GpuModeProcessor *processor = new GpuModeProcessor();

    ProcessorContext *context;
    createContext(&context, 512);

    // prepare data
    for (auto i = 0; i < 512; i++) {
        context->in[i].x = cos(TWO_PI * i / 512);
        context->in[i].y = sin(2./3. * TWO_PI * i / 512 + 45 * DEG_TO_RAD);
    }

    cout << "Context length = " << context->length << endl;

    // ProcessorContext *context = (ProcessorContext *)malloc(sizeof(ProcessorContext));
    // context->length = 512;

    processor->process(context);

    // get the data out
    for (auto i = 0; i < 512; i++) {
        cout << "out[" << i << "]: (" << context->out[i].x << ", " << context->out[i].y << ")" << endl;
    }

    freeContext(context);
    free(processor);
}
