#ifndef GPUMODE_H
#define GPUMODE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>
#include "mode.h"

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

    void preprocess(int index, int subloopindex);
    void postprocess(int index, int subloopindex);
    void runFFT();
    void complexRotate(int subloopindex);

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
private:

    cufftHandle fft_plan;
    int cfg_numBufferedFFTs;
    double averagedelay, nearestsampletime, starttime, lofreq, walltimesecs, fracwalltime, fftcentre, d0, d1, d2, fraclooffset;
    f32 fracsampleerror;
    int status, count, nearestsample, integerdelay, intwalltime;
    f32 *currentstepchannelfreqs;
    f32 *currentsubchannelfreqs;
    int indices[10];
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
