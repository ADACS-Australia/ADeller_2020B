#define NOT_SUPPORTED(x) { std::cerr << "Whoops, we don't support this on the GPU: " << x << std::endl; exit(1); }

#include "gpumode.cuh"
#include "alert.h"
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>
#include <cufftXt.h>

#include "gpumode_kernels.cuh"
#include <chrono>
#include <omp.h>
#include <thread>

using namespace std::chrono;

GPUMode::GPUMode(Configuration *conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend,
                 int gsamples, int nrecordedfreqs, double recordedbw, double *recordedfreqclkoffs,
                 double *recordedfreqclkoffsdelta, double *recordedfreqphaseoffs, double *recordedfreqlooffs,
                 int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling,
                 Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular,
                 int fringerotorder, int arraystridelen, bool cacorrs, double bclock) :
        Mode(conf, confindex, dsindex, recordedbandchan, chanstoavg, bpersend, gsamples, nrecordedfreqs, recordedbw,
             recordedfreqclkoffs, recordedfreqclkoffsdelta, recordedfreqphaseoffs, recordedfreqlooffs, nrecordedbands,
             nzoombands, nbits, sampling, tcomplex, unpacksamp, fbank, linear2circular, fringerotorder, arraystridelen,
             cacorrs, bclock), estimatedbytes_gpu(0) {

    auto start = high_resolution_clock::now();

    cfg_numBufferedFFTs = config->getNumBufferedFFTs(confindex);
    this->unpackedarrays_elem_count = unpacksamples;

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties( &prop, 0));

    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;

    this->complexunpacked_gpu = gpu_malloc<cuFloatComplex>(this->fftchannels * cfg_numBufferedFFTs * numrecordedbands);
    this->estimatedbytes_gpu += sizeof(cuFloatComplex) * this->fftchannels * cfg_numBufferedFFTs * numrecordedbands;

    this->fftd_gpu = gpu_malloc<cuFloatComplex>(this->fftchannels * cfg_numBufferedFFTs * numrecordedbands);
    this->fftd_gpu_out = new cf32[this->fftchannels * cfg_numBufferedFFTs * numrecordedbands];
    this->estimatedbytes_gpu += sizeof(cuFloatComplex) * this->fftchannels * cfg_numBufferedFFTs * numrecordedbands;

    this->unpackedarrays_cpu = new float *[numrecordedbands * cfg_numBufferedFFTs];
    float *big_array = new float[unpackedarrays_elem_count * numrecordedbands * cfg_numBufferedFFTs];
    for (int j = 0; j < cfg_numBufferedFFTs; j++) {
        for (size_t i = 0; i < numrecordedbands; i++) {
            this->unpackedarrays_cpu[(j * numrecordedbands) + i] =
                    big_array + (((j * numrecordedbands) + i) * unpackedarrays_elem_count);
        }
    }

    this->unpackedarrays_gpu = new float*[numrecordedbands * cfg_numBufferedFFTs];
    this->estimatedbytes += sizeof(float *) * numrecordedbands;

    big_array = nullptr;
    checkCuda(cudaMalloc(&big_array, sizeof(float) * unpackedarrays_elem_count * numrecordedbands * cfg_numBufferedFFTs));
    cudaMemset(&big_array, 0, sizeof(float) * unpackedarrays_elem_count * numrecordedbands * cfg_numBufferedFFTs);
    this->estimatedbytes_gpu += sizeof(float) * this->unpackedarrays_elem_count * numrecordedbands * cfg_numBufferedFFTs;
    for (int j = 0; j < cfg_numBufferedFFTs; j++) {
        for (size_t i = 0; i < numrecordedbands; i++) {
            this->unpackedarrays_gpu[(j * numrecordedbands) + i] =
                    big_array + (((j * numrecordedbands) + i) * unpackedarrays_elem_count);
        }
    }

    fracsamprotatorA_array = new cf32 *[cfg_numBufferedFFTs];
    for (int j = 0; j < cfg_numBufferedFFTs; j++) {
        fracsamprotatorA_array[j] = vectorAlloc_cf32(recordedbandchannels);
    }

    sampleIndexes = new int[cfg_numBufferedFFTs];
    validSamples = new bool[cfg_numBufferedFFTs];

    checkCuda(cudaMalloc(&gSampleIndexes, sizeof(int) * cfg_numBufferedFFTs));
    checkCuda(cudaMalloc(&gValidSamples, sizeof(bool) * cfg_numBufferedFFTs));
    checkCuda(cudaMalloc(&gUnpackedArraysGpu, sizeof(float*) * numrecordedbands * cfg_numBufferedFFTs));

    // Register host ram used to copy data to gpu
    checkCuda(cudaHostRegister(this->unpackedarrays_cpu[0], sizeof(float) * unpackedarrays_elem_count * numrecordedbands * cfg_numBufferedFFTs, cudaHostRegisterPortable));
    checkCuda(cudaHostRegister(sampleIndexes, sizeof(int) * cfg_numBufferedFFTs, cudaHostRegisterPortable));
    checkCuda(cudaHostRegister(validSamples, sizeof(bool) * cfg_numBufferedFFTs, cudaHostRegisterPortable));
    checkCuda(cudaHostRegister(fftd_gpu_out, sizeof(cf32) * this->fftchannels * cfg_numBufferedFFTs * numrecordedbands, cudaHostRegisterPortable));

    checkCuda(cudaStreamCreate(&cuStream));

    // TODO: PWC: allocations for complex

    int n[] = {this->fftchannels};
    int istride = 1;
    int ostride = 1;
    int idist = this->fftchannels;
    int odist = this->fftchannels;

    int inembed[] = {0};
    int onembed[] = {0};

    checkCufft(
            cufftPlanMany(
                    &this->fft_plan,
                    1,
                    (int *) &n,
                    (int *) &inembed,
                    istride,
                    idist,
                    (int *) &onembed,
                    ostride,
                    odist,
                    CUFFT_C2C,
                    numrecordedbands * cfg_numBufferedFFTs
            )
    );
    checkCufft(cufftSetStream(fft_plan, cuStream));

    // littleA/B
    checkCuda(cudaMalloc(&gInterpolator, sizeof(double) * 3));
    checkCuda(cudaHostRegister(interpolator, sizeof(double) * 3, cudaHostRegisterPortable));

    // precalc
    fracSampleError = new float[cfg_numBufferedFFTs];
    nearestSample = new int[cfg_numBufferedFFTs];

    checkCuda(cudaMalloc(&gFracSampleError, sizeof(float) * cfg_numBufferedFFTs));

    checkCuda(cudaHostRegister(fracSampleError, sizeof(float) * cfg_numBufferedFFTs, cudaHostRegisterPortable));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "GPUMode(): " << duration.count() << endl;
}

unsigned long long avg_unpack;
unsigned long long avg_rotate;
unsigned long long avg_fft;
unsigned long long avg_postprocess;
unsigned long long processing_time;

int calls = 0;

GPUMode::~GPUMode() {
    auto start = high_resolution_clock::now();

    checkCuda(cudaHostUnregister(this->unpackedarrays_cpu[0]));
    checkCuda(cudaHostUnregister(sampleIndexes));
    checkCuda(cudaHostUnregister(validSamples));
    checkCuda(cudaHostUnregister(fftd_gpu_out));

    checkCuda(cudaFree(this->complexunpacked_gpu));
    checkCuda(cudaFree(this->fftd_gpu));

    checkCuda(cudaFree(gSampleIndexes));
    checkCuda(cudaFree(gValidSamples));

    // Allocated on the GPU as one big array so we don't need to free them all
    checkCuda(cudaFree(this->unpackedarrays_gpu[0]));
    delete[] this->unpackedarrays_gpu;
    delete[] this->fftd_gpu_out;
    // TODO: PWC: dealloctions for complex

    cufftDestroy(this->fft_plan);

    checkCuda(cudaStreamDestroy(cuStream));

    // precalc
    delete fracSampleError;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "~GPUMode(): " << duration.count() << endl;

    cout << "Average unpack: " << avg_unpack / calls << endl;
    cout << "Average rotate: " << avg_rotate / calls << endl;
    cout << "Average fft: " << avg_fft / calls << endl;
    cout << "Average postprocess: " << avg_postprocess / calls << endl;
    cout << "Actual time processing (seconds): " << (double) processing_time / 1000. / 1000. << endl;
}

int GPUMode::process_gpu(int fftloop, int numBufferedFFTs, int startblock,
                         int numblocks)  //frac sample error is in microseconds
{
    auto begin_time = high_resolution_clock::now();

    calls += 1;
//    std::cout << "Doing the thing. fftloop: " << fftloop << ", numBufferedFFTs: " << numBufferedFFTs << ", numblocks: " << numblocks << ", startblock: " << startblock << std::endl;

    // Sanity checks
    if (perbandweights) {
        NOT_SUPPORTED("per band weights");
    }

    if (!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) {
        NOT_SUPPORTED("DPhaseCal");
    }

    if (fringerotationorder != 1) { // linear only
        NOT_SUPPORTED("fringerotationorder = " + to_string(fringerotationorder));
    }

    if (1 != numrecordedfreqs) {
        NOT_SUPPORTED("a value for 'numrecordedfreqs' other than 1");
    }

    if (usedouble) {
        NOT_SUPPORTED("usedouble branch");
    }

    if (recordedfreqlooffsets[0] > 0.0 || recordedfreqlooffsets[0] < 0.0) {
        NOT_SUPPORTED("lo offsets");
    }

    if (usecomplex && usedouble) {
        NOT_SUPPORTED("complex double-sideband data");
    } else if (usecomplex) {
        NOT_SUPPORTED("complex data");
    }

    if (deltapoloffsets) {
        NOT_SUPPORTED("deltapoloffsets");
    }

    if (config->getDRecordedLowerSideband(configindex, datastreamindex, 0)) {
        NOT_SUPPORTED("lower sideband");
    }

    if (dumpkurtosis) {
        NOT_SUPPORTED("dump_kurtosis branch");
    }

    if (linear2circular) {
        NOT_SUPPORTED("linear to circular polarisation conversion");
    } else if (phasepoloffset) {
        NOT_SUPPORTED("phase polarisation offset");
    }

    // Update the interpolator
    checkCuda(cudaMemcpyAsync(gInterpolator, interpolator, sizeof(double) * 3, cudaMemcpyHostToDevice, cuStream));

    auto start = high_resolution_clock::now();
    calculatePre_cpu(fftloop, numBufferedFFTs, startblock, numblocks);

    // First unpack all the data
    for (int subloopindex = 0; subloopindex < numBufferedFFTs; subloopindex++) {
        int i = fftloop * numBufferedFFTs + subloopindex + startblock;
        if (i >= startblock + numblocks)
            break; // may not have to fully complete last fftloop

        process_unpack(i, subloopindex);
    }

    // Copy the data to the gpu
    checkCuda(cudaMemcpyAsync(this->unpackedarrays_gpu[0], this->unpackedarrays_cpu[0], sizeof(float) * unpackedarrays_elem_count * numrecordedbands * cfg_numBufferedFFTs, cudaMemcpyHostToDevice, cuStream));

    // We need to copy the sample indexes to the gpu
    checkCuda(cudaMemcpyAsync(gSampleIndexes, sampleIndexes, sizeof(int) * cfg_numBufferedFFTs, cudaMemcpyHostToDevice, cuStream));
    checkCuda(cudaMemcpyAsync(gValidSamples, validSamples, sizeof(bool) * cfg_numBufferedFFTs, cudaMemcpyHostToDevice, cuStream));
    checkCuda(cudaMemcpyAsync(gUnpackedArraysGpu, unpackedarrays_gpu, sizeof(float*) * numrecordedbands * cfg_numBufferedFFTs, cudaMemcpyHostToDevice, cuStream));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "unpack: " << duration.count() << endl;
    avg_unpack += duration.count();

    start = high_resolution_clock::now();

    // Run the rotator
    complexRotate(fftloop, numBufferedFFTs, startblock, numblocks);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "rotate: " << duration.count() << endl;
    avg_rotate += duration.count();

    start = high_resolution_clock::now();
    // Actually run the FFT
    runFFT();

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "fft: " << duration.count() << endl;
    avg_fft += duration.count();

    start = high_resolution_clock::now();

    rotateResults(fftloop, numBufferedFFTs, startblock, numblocks);

    int numfftsprocessed = 0;
    // Do stuff with the FFT results
    for (; numfftsprocessed < numBufferedFFTs; numfftsprocessed++) {
        int i = fftloop * numBufferedFFTs + numfftsprocessed + startblock;
        if (i >= startblock + numblocks)
            break; // may not have to fully complete last fftloop

        postprocess(i, numfftsprocessed);
    }

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "postprocess: " << duration.count() << endl;
    avg_postprocess += duration.count();

    processing_time += duration_cast<microseconds>(stop - begin_time).count();

    return numfftsprocessed;
}

bool GPUMode::is_dataweight_valid(int subloopindex) {
    int status;

    if (!(dataweight[subloopindex] > 0.0)) {
        for (int i = 0; i < numrecordedbands; i++) {
            status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
            status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
        }
        return false;
    }

    return true;
}

bool GPUMode::is_data_valid(int index, int subloopindex) {
    int status;

    // Check the data is valid for this index
    if ((datalengthbytes <= 1) || (offsetseconds == INVALID_SUBINT) ||
        (((validflags[index / FLAGS_PER_INT] >> (index % FLAGS_PER_INT)) & 0x01) == 0)) {
//        std::cerr << "to M::p_g; we are in the weird place with the datalengthbytes" << std::endl;
//        std::cerr << "to M::p_g; numrecorededbands = " << numrecordedbands << std::endl;
        for (int i = 0; i < numrecordedbands; i++) {
            status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
            status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
        }
//        cerr << "Mode for DS " << datastreamindex << " is bailing out of index " << index << "/" << subloopindex << " which is scan " << currentscan << ", sec " << offsetseconds << ", ns " << offsetns << " because datalengthbytes is " << datalengthbytes << " and validflag was " << ((validflags[index/FLAGS_PER_INT] >> (index%FLAGS_PER_INT)) & 0x01) << endl;
        return false; //don't process crap data
    }

    // Check that the nearest sample is valid
    if (nearestSample[subloopindex] < -1 ||
        (((nearestSample[subloopindex] + fftchannels) / samplesperblock) * bytesperblocknumerator) / bytesperblockdenominator >
        datalengthbytes) {
//        std::cerr << "to M::p_g; we are in the 'crap data' branch" << std::endl;
//        cerror << startl << "MODE error for datastream " << datastreamindex
//               << " - trying to process data outside range - aborting!!! nearest sample was " << nearestSample[subloopindex]
//               << ", the max bytes should be " << datalengthbytes << " and hence last sample should be "
//               << (datalengthbytes * bytesperblockdenominator) / (bytesperblocknumerator * samplesperblock)
//               << " (fftchannels is " << fftchannels << "), offsetseconds was " << offsetseconds << ", offsetns was "
//               << offsetns << ", index was " << index << ", average delay was " << nearestSample[subloopindex] << ", datasec was "
//               << datasec << ", datans was " << datans << ", fftstartmicrosec was " << fftstartmicrosec << endl;
        for (int i = 0; i < numrecordedbands; i++) {
            status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
            status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
        }
        return false;
    }

    return true;
}

void GPUMode::process_unpack(int index, int subloopindex) {
    static int nth_call = 0;
    ++nth_call;

    // since these data weights can be retreived after this processing ends, reset them to a default of zero in case they don't get updated
    dataweight[subloopindex] = 0.0;

    if (!is_data_valid(index, subloopindex)) {
        validSamples[subloopindex] = false;
        return;
    }

    validSamples[subloopindex] = true;

    double fftcentre = index + 0.5;
    double averagedelay = interpolator[0] * fftcentre * fftcentre + interpolator[1] * fftcentre + interpolator[2];

    double fftstartmicrosec = index * fftchannels * sampletime;

    double starttime = (offsetseconds - datasec) * 1000000.0 +
                       (static_cast<long long>(offsetns) - static_cast<long long>(datans)) / 1000.0 + fftstartmicrosec -
                       averagedelay;

    int nearestsample = int(starttime / sampletime + 0.5);

    if (nearestsample == -1) {
        nearestsample = 0;
        dataweight[subloopindex] = unpack(nearestsample, subloopindex);
    } else if (nearestsample < unpackstartsamples || nearestsample > unpackstartsamples + unpacksamples - fftchannels)
        //need to unpack more data
        dataweight[subloopindex] = unpack(nearestsample, subloopindex);

    sampleIndexes[subloopindex] = nearestsample - unpackstartsamples;

    if (!is_dataweight_valid(subloopindex)) {
        validSamples[subloopindex] = false;
    }
}

void GPUMode::calculatePre_cpu(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {
    int startIndex = fftloop * numBufferedFFTs + startblock;
    int endIndex = startblock + numblocks;

    for (int subloopindex = 0; subloopindex < numBufferedFFTs; subloopindex++) {
        int index = startIndex + subloopindex;
        if (index >= endIndex)
            break; // may not have to fully complete last fftloop

        double fftcentre = index + 0.5;
        double averagedelay = interpolator[0] * fftcentre * fftcentre + interpolator[1] * fftcentre + interpolator[2];
        double fftstartmicrosec = index * fftchannels * sampletime; //CHRIS CHECK
        double starttime = (offsetseconds - datasec) * 1000000.0 +
                           (static_cast<long long>(offsetns) - static_cast<long long>(datans)) / 1000.0 + fftstartmicrosec -
                           averagedelay;
        nearestSample[subloopindex] = int(starttime / sampletime + 0.5);


        double nearestsampletime = nearestSample[subloopindex] * sampletime;
        fracSampleError[subloopindex] = float(starttime - nearestsampletime);
    }

    checkCuda(cudaMemcpyAsync(gFracSampleError, fracSampleError, sizeof(float) * cfg_numBufferedFFTs, cudaMemcpyHostToDevice, cuStream));
}

void GPUMode::preprocess(int subloopindex) {
    int status;

    if (!validSamples[subloopindex]) {
        return;
    }

    // Do the main work here
    // Loop over each frequency and to the fringe rotation and FFT of the data

    //updated so that Nyquist channel is not accumulated for either USB or LSB data
    //and is excised entirely, so both USB and LSB data start at the same place (no sidebandoffset)
    f32* currentstepchannelfreqs = stepchannelfreqs;
    f32* currentsubchannelfreqs = subchannelfreqs;
    if (config->getDRecordedLowerSideband(configindex, datastreamindex, 0)) {
        currentstepchannelfreqs = lsbstepchannelfreqs;
    }

    // For double-sideband data, the LO frequency is at the centre of the band, not the band edge

    // OK, now let's put some actual GPU in here

/* The actual calculation that is going on for the linear case is as follows:

 Calculate complexrotator[j]  (for j = 0 to fftchanels-1) as:

 complexrotator[j] = exp( 2 pi i * (A*j + B) )

 where:

 A = a*lofreq/fftchannels - sampletime*1.0e-6*recordedfreqlooffsets[i]
 B = b*lofreq/fftchannels + fraclofreq*integerdelay - recordedfreqlooffsets[i]*fracwalltime - fraclooffset*intwalltime

 And a, b are computed outside the recordedfreq loop (variable i)
*/

/* Creating a fractional sample rotation array
 *  The actual calculation being performed is as follows:
 *  Assume we know the frequency of every FFT output channel, and it is stored in an array of length fftchannels, called channelfreq
 *  then for every frequency subband f (in the range 0 … numrecordedfreqs), calculate the slope as:
 *  A = fracsampleerror - recordedfreqclockoffsets[f] + recordedfreqclockoffsetsdelta[f]/2
 *  (for the second polarisation, a is identical except subtracting recordedfreqclockoffsetsdelta[f]/2)
 * then calculate complexrotator[j]  (for j = 0 to fftchannels-1) as:
 * complexrotator[j] = exp( 2 pi i * (A*fftchannels[j]) )
 *
 * So how is fftchannels calculated? For “regular data” it is as follows (for j = 0 to fftchannels-1)
 * fftchannels[j] = recordedbandwidth * j / fftchannels
 * For lower sideband data it is:
 * fftchannels[j] = -recordedbandwidth * j / fftchannels
 * For double sideband data it is:
 * fftchannels[j] = recordedbandwidth * j / fftchannels - recordedbandwidth/2.0
 */

    // Note recordedfreqclockoffsetsdata will usually be zero, but avoiding if statement
    status = vectorMulC_f32(currentsubchannelfreqs,
                            fracSampleError[subloopindex] - recordedfreqclockoffsets[0] + recordedfreqclockoffsetsdelta[0] / 2,
                            subfracsamparg, arraystridelength);
    if (status != vecNoErr) {
        csevere << startl << "Error in frac sample correction, arg generation (sub)!!!" << status << endl;
        exit(1);
    }
    status = vectorMulC_f32(currentstepchannelfreqs,
                            fracSampleError[subloopindex] - recordedfreqclockoffsets[0] + recordedfreqclockoffsetsdelta[0] / 2,
                            stepfracsamparg, numfracstrides / 2);
    if (status != vecNoErr)
        csevere << startl << "Error in frac sample correction, arg generation (step)!!!" << status << endl;

    //create the fractional sample correction array
    status = vectorSinCos_f32(subfracsamparg, subfracsampsin, subfracsampcos, arraystridelength);
    if (status != vecNoErr)
        csevere << startl << "Error in frac sample correction, sin/cos (sub)!!!" << status << endl;
    status = vectorSinCos_f32(stepfracsamparg, stepfracsampsin, stepfracsampcos, numfracstrides / 2);
    if (status != vecNoErr)
        csevere << startl << "Error in frac sample correction, sin/cos (sub)!!!" << status << endl;
    status = vectorRealToComplex_f32(subfracsampcos, subfracsampsin, fracsamprotatorA_array[subloopindex], arraystridelength);
    if (status != vecNoErr)
        csevere << startl << "Error in frac sample correction, real to complex (sub)!!!" << status << endl;
    status = vectorRealToComplex_f32(stepfracsampcos, stepfracsampsin, stepfracsampcplx, numfracstrides / 2);
    if (status != vecNoErr)
        csevere << startl << "Error in frac sample correction, real to complex (step)!!!" << status << endl;
    for (int j = 1; j < numfracstrides / 2; j++) {
        status = vectorMulC_cf32(fracsamprotatorA_array[subloopindex], stepfracsampcplx[j], &(fracsamprotatorA_array[subloopindex][j * arraystridelength]),
                                 arraystridelength);
        if (status != vecNoErr)
            csevere << startl << "Error doing the time-saving complex multiplication in frac sample correction!!!"
                    << endl;
    }

    // now do the first arraystridelength elements (which are different from fracsampptr1 for LSB case)
    status = vectorMulC_cf32_I(stepfracsampcplx[0], fracsamprotatorA_array[subloopindex], arraystridelength);
    if (status != vecNoErr)
        csevere << startl
                << "Error doing the first bit of the time-saving complex multiplication in frac sample correction!!!"
                << endl;


}

__global__ void _gpu_complexrotatorMultiply(
        cuFloatComplex* const dest,
        float **const src,
        const double* const interpolator,
        const int* const sampleIndexes,
        const bool* const validSamples,
        double lofreq,
        double sampletime,
        double recordedfreqlooffset,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels
    ) {
    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the numrecordedbands index [0 .. numrecordedbands]
    // threadIdx.y in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the numrecordedbands size
    // blockDim.y in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    // Check if this subloopindex is valid
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) {
        // Not valid, so don't do anything
        return;
    }

    // Check if we should bother processing this sample
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        // May not have to fully complete last fftloop, drop out
        return;
    }

    const size_t bandindex = threadIdx.x;
    const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;
    const size_t numrecordedbands = blockDim.x;

    if (channelindex >= fftchannels) {
        return;
    }

    // Calculate the destination index
    const size_t destIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

    // Calculate the source index and get the source value
    const size_t srcIndex = (subloopindex * numrecordedbands) + bandindex;
    const float srcVal = src[srcIndex][sampleIndexes[subloopindex] + channelindex];

    // Calculate littleA/B
    double d0 = interpolator[0] * index * index + interpolator[1] * index + interpolator[2];
    double d1 = interpolator[0] * (index + 0.5) * (index + 0.5) + interpolator[1] * (index + 0.5) + interpolator[2];
    double d2 = interpolator[0] * (index + 1) * (index + 1) + interpolator[1] * (index + 1) + interpolator[2];

    double a = d2 - d0;
    double b = d0 + (d1 - (a * 0.5 + d0)) / 3.0;
    
    // Calculate BigA/B
    double bigAval = a * lofreq / fftchannels - sampletime * 1.e-6 * recordedfreqlooffset;
    double bigBval = b * lofreq;

    // Calculate
    double bigB_reduced = bigBval - int(bigBval);
    double exponent = (bigAval * channelindex + bigB_reduced);
    exponent -= int(exponent);
    cuFloatComplex cr;
    sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
    cuFloatComplex c = make_cuFloatComplex(srcVal, 0.f);
    dest[destIndex] = cuCmulf(c, cr);
}

void GPUMode::complexRotate(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {

    // At this point we have
    // * Unpacked data on GPU
    // * Output buffer on GPU ready to go
    // * Sample indexes in the unpacked data
    // * BigA and BigB
    // * Which samples are valid - ie that we need to operate on

    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))
    size_t fftchannels_block;
    size_t fftchannels_grid;

    size_t divisor = cudaMaxThreadsPerBlock / numrecordedbands;
    if (fftchannels > divisor) {
        fftchannels_block = divisor;
        fftchannels_grid = (fftchannels / divisor);

        if (fftchannels % divisor != 0) {
            fftchannels_grid++;
        }
    } else {
        fftchannels_block = fftchannels;
        fftchannels_grid = 1;
    }

    _gpu_complexrotatorMultiply<<<dim3(numBufferedFFTs, fftchannels_grid), dim3(numrecordedbands, fftchannels_block), 0, cuStream>>>
    (
             complexunpacked_gpu,
             gUnpackedArraysGpu,
             gInterpolator,
             sampleIndexes,
             validSamples,
             config->getDRecordedFreq(configindex, datastreamindex, 0),
             sampletime,
             recordedfreqlooffsets[0],
             fftloop,
             startblock,
             numblocks,
             fftchannels
     );
}

__global__ void _gpu_resultsrotatorMultiply(
        cuFloatComplex* const srcdest,
        const float* const fracSampleError,
        const bool* const validSamples,
        const double recordedbandwidth,
        double recordedfreqclockoffset,
        double recordedfreqclockoffsetdelta,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels
    ) {
    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the numrecordedbands index [0 .. numrecordedbands]
    // threadIdx.y in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the numrecordedbands size
    // blockDim.y in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    // Check if this subloopindex is valid
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) {
        // Not valid, so don't do anything
        return;
    }

    // Check if we should bother processing this sample
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        // May not have to fully complete last fftloop, drop out
        return;
    }

    const size_t bandindex = threadIdx.x;
    const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;
    const size_t numrecordedbands = blockDim.x;

    if (channelindex >= fftchannels) {
        return;
    }

    // Calculate the destination index
    const size_t dataIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

    // Get fracsampleerror - recordedfreqclockoffsets[f] + recordedfreqclockoffsetsdelta[f]/2
    double bigAval = fracSampleError[subloopindex] - recordedfreqclockoffset + recordedfreqclockoffsetdelta/2;

    // Generate fftchannels[j] = recordedbandwidth * j / fftchannels
    double subFreq = recordedbandwidth * channelindex / fftchannels;

    // Calculate
    double exponent = bigAval * subFreq;
    exponent -= int(exponent);
    cuFloatComplex cr;
    sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
    srcdest[dataIndex] = cuCmulf(srcdest[dataIndex], cr);
}

void GPUMode::rotateResults(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {
    // At this point we have
    // * FFT results on GPU
    // * Rotator BigA
    // * subchannelfreqs
    // * Which samples are valid - ie that we need to operate on

    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))
    size_t fftchannels_block;
    size_t fftchannels_grid;

    size_t divisor = cudaMaxThreadsPerBlock / numrecordedbands;
    if (fftchannels > divisor) {
        fftchannels_block = divisor;
        fftchannels_grid = (fftchannels / divisor);

        if (fftchannels % divisor != 0) {
            fftchannels_grid++;
        }
    } else {
        fftchannels_block = fftchannels;
        fftchannels_grid = 1;
    }

    _gpu_resultsrotatorMultiply<<<dim3(numBufferedFFTs, fftchannels_grid), dim3(numrecordedbands, fftchannels_block), 0, cuStream>>>
            (
                    fftd_gpu,
                    gFracSampleError,
                    validSamples,
                    recordedbandwidth,
                    recordedfreqclockoffsets[0],
                    recordedfreqclockoffsetsdelta[0],
                    fftloop,
                    startblock,
                    numblocks,
                    fftchannels
            );

    checkCuda(cudaMemcpyAsync(fftd_gpu_out, this->fftd_gpu,
                              sizeof(cuFloatComplex) * this->fftchannels * numrecordedbands * cfg_numBufferedFFTs,
                              cudaMemcpyDeviceToHost, cuStream));

    checkCuda(cudaStreamSynchronize(cuStream));
}

void GPUMode::postprocess(int index, int subloopindex) {
    int status;
    int count = 0;
    int indices[10];

    if (!validSamples[subloopindex]) {
        return;
    }

    // PWCR numrecordedbands = 2 for the test; but e.g. 8 is very realistical
    // Loop over all recorded bands looking for the matching frequency we should be dealing with
    for (int j = 0; j < numrecordedbands; j++) {
        if (config->matchingRecordedBand(configindex, datastreamindex, 0, j)) {
            indices[count++] = j;

            // For upper sideband bands, normally just need to copy the fftd channels.
            // However for complex double upper sideband, the two halves of the frequency space are swapped, so they need to be swapped back
            status = vectorCopy_cf32(&fftd_gpu_out[(subloopindex * fftchannels * numrecordedbands) + (j * fftchannels)],
                                     fftoutputs[j][subloopindex],
                                     recordedbandchannels);

            if (status != vecNoErr)
                csevere << startl << "Error copying FFT results!!!" << endl;


            // At this point in the code the array fftoutputs[j] contains complex-valued voltage spectra with the following properties:
            //
            // 1. The zero element corresponds to the lowest sky frequency.  That is:
            //    fftoutputs[j][0] = Local Oscillator Frequency              (for Upper Sideband)
            //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth  (for Lower Sideband)
            //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth  (for Complex Lower Sideband)
            //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Upper Sideband)
            //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Lower Sideband)
            //
            // 2. The frequency increases monotonically with index
            //
            // 3. The last element of the array corresponds to the highest sky frequency minus the spectral resolution.
            //    (i.e., the first element beyond the array bound corresponds to the highest sky frequency)


//            //do the frac sample correct (+ phase shifting if applicable, + fringe rotate if its post-f)
//            if (deltapoloffsets == false || config->getDRecordedBandPol(configindex, datastreamindex, j) == 'R') {
//                status = vectorMul_cf32_I(fracsamprotatorA_array[subloopindex], fftoutputs[j][subloopindex], recordedbandchannels);
//            } else {
//                NOT_SUPPORTED("fracsamplerotatorB");
//            }

//            if (status != vecNoErr)
//                csevere << startl << "Error in application of frac sample correction!!!" << status << endl;

            //do the conjugation
            status = vectorConj_cf32(fftoutputs[j][subloopindex], conjfftoutputs[j][subloopindex],
                                     recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error in conjugate!!!" << status << endl;

            if (!linear2circular) {
                //do the autocorrelation (skipping Nyquist channel)
                status = vectorAddProduct_cf32(fftoutputs[j][subloopindex], conjfftoutputs[j][subloopindex],
                                               autocorrelations[0][j], recordedbandchannels);
                if (status != vecNoErr)
                    csevere << startl << "Error in autocorrelation!!!" << status << endl;

                //store the weight for the autocorrelations
                if (perbandweights) {
                    weights[0][j] += perbandweights[subloopindex][j];
                } else {
                    weights[0][j] += dataweight[subloopindex];
                }
            }
        }
    }

    if (count > 1) {
        //if we need to, do the cross-polar autocorrelations
        if (calccrosspolautocorrs) {
            status = vectorAddProduct_cf32(fftoutputs[indices[0]][subloopindex],
                                           conjfftoutputs[indices[1]][subloopindex],
                                           autocorrelations[1][indices[0]],
                                           recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error in cross-polar autocorrelation!!!" << status << endl;
            status = vectorAddProduct_cf32(fftoutputs[indices[1]][subloopindex],
                                           conjfftoutputs[indices[0]][subloopindex],
                                           autocorrelations[1][indices[1]],
                                           recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error in cross-polar autocorrelation!!!" << status << endl;

            //store the weights
            if (perbandweights) {
                weights[1][indices[0]] +=
                        perbandweights[subloopindex][indices[0]] * perbandweights[subloopindex][indices[1]];
                weights[1][indices[1]] +=
                        perbandweights[subloopindex][indices[0]] * perbandweights[subloopindex][indices[1]];
            } else {
                weights[1][indices[0]] += dataweight[subloopindex];
                weights[1][indices[1]] += dataweight[subloopindex];
            }
        }
    }
}

void GPUMode::runFFT() {
    checkCufft(cufftExecC2C(this->fft_plan, this->complexunpacked_gpu, fftd_gpu, CUFFT_FORWARD));
}

__global__ void _gpu_inPlaceMultiply_cf(const cuFloatComplex *const src, cuFloatComplex *const dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = cuCmulf(dst[idx], src[idx]);
}

void gpu_inPlaceMultiply_cf(const cuFloatComplex *const dst, cuFloatComplex *const bydst, const size_t len) {
    _gpu_inPlaceMultiply_cf<<<1, len>>>(dst, bydst);
}

// Copy from host to device, converting from float to cuFloatComplex
// (initialising all imaginary parts as zero) as you go
void gpu_host2DevRtoC(cuFloatComplex *const dst, const float *const src, const size_t len) {
    checkCuda(cudaMemset(dst, 0x0, len * sizeof(cuFloatComplex)));
    checkCuda(cudaMemcpy2D(dst, sizeof(cuFloatComplex), src, sizeof(float), sizeof(float), len,
                           cudaMemcpyHostToDevice));
}

void *gpu_malloc(const size_t amt) {
    void *rv;
    checkCuda(cudaMalloc(&rv, amt));
    return rv;
}

__global__ void _cudaMul_f64_many(double *const src, double *const dest, double *const a, int vecElems, int numVecs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= vecElems * numVecs)
        return;

    dest[idx] = src[idx % vecElems] * a[idx / vecElems];
}

void cudaMul_f64_many(double *const src, double *const dest, double *const a, int vecElems, int numVecs, int cudaMaxThreadsPerBlock, cudaStream_t cuStream) {
    size_t elems_block;
    size_t elems_grid;

    size_t elems = vecElems * numVecs;

    if (elems > cudaMaxThreadsPerBlock) {
        elems_block = cudaMaxThreadsPerBlock;
        elems_grid = (elems / cudaMaxThreadsPerBlock);

        if (elems % cudaMaxThreadsPerBlock != 0) {
            elems_grid++;
        }
    } else {
        elems_block = elems;
        elems_grid = 1;
    }

    _cudaMul_f64_many<<<elems_grid, elems_block, 0, cuStream>>>(src, dest, a, vecElems, numVecs);
}


