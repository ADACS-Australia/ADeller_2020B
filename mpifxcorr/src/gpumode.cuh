#ifndef GPUMODE_H
#define GPUMODE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>
#include "mode.h"
#include <mutex>

class Configuration;

class GPUMode : public Mode {
public:
    GPUMode(Configuration *conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend,
            int gsamples, int nrecordedfreqs, double recordedbw, double *recordedfreqclkoffs,
            double *recordedfreqclkoffsdelta, double *recordedfreqphaseoffs, double *recordedfreqlooffs,
            int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling,
            Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular, int fringerotorder,
            int arraystridelen, bool cacorrs, double bclock);

    ~GPUMode() override;

    int process_gpu(int fftloop, int numBufferedFFTs, int startblock,
                    int numblocks) override;  //frac sample error is in microseconds

    void process_unpack(int index, int subloopindex);
    void preprocess(int index, int subloopindex);
    void postprocess(int index, int subloopindex);
    void runFFT();
    void complexRotate(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void postprocess_gpu(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void calculateLittleAB(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void calculateLittleABCleanup();

protected:
    float **unpackedarrays_gpu;
    float **unpackedarrays_cpu;

    cuFloatComplex *complexunpacked_gpu;
    cuFloatComplex *fftd_gpu;
    cf32* fftd_gpu_out;

    size_t estimatedbytes_gpu;

    cuFloatComplex *complexrotator_gpu;

    // Remember how long the 'unpackedarrays' are -- norally this would be
    // 'unpacksamples' but e.g. the Mk5Mode implementation overwrites that
    size_t unpackedarrays_elem_count;

    cf32** fracsamprotatorA_array;

    double* bigA_d;
    double* bigB_d;
    int* sampleIndexes;
    bool* validSamples;

    double *gBigA, *gBigB;
    int *gSampleIndexes;
    bool *gValidSamples;
    float** gUnpackedArraysGpu;

    cudaStream_t cuStream;

    // calculateLittleAB
    double* gInterpolator;
    double* littleA;
    double* littleB;
    double* gLittleA;
    double* gLittleB;
    int* integerDelay;
    int* gIntegerDelay;

private:

    cufftHandle fft_plan;
    int cfg_numBufferedFFTs;

    bool is_dataweight_valid(int subloopindex);
    bool is_data_valid(int index, int subloopindex);
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
