#include "gpumode.cuh"
#include "alert.h"
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>
#include <cufftXt.h>

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
    unpackedarrays_elem_count = unpacksamples;

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties( &prop, 0));

    checkCuda(cudaStreamCreate(&cuStream));

    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;

    complexunpacked_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, true);
    estimatedbytes_gpu += complexunpacked_gpu->size();

    fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, true);
    estimatedbytes_gpu += fftd_gpu->size();

    conj_fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, true);
    estimatedbytes_gpu += conj_fftd_gpu->size();

    temp_autocorrelations_gpu = new GpuMemHelper<cuFloatComplex>(numrecordedbands * recordedbandchannels * 3, cuStream);
    estimatedbytes_gpu += temp_autocorrelations_gpu->size();

    unpackedarrays_gpu = new GpuMemHelper<float*>(numrecordedbands * cfg_numBufferedFFTs, cuStream);
    unpackeddata_gpu = new GpuMemHelper<float>(unpackedarrays_elem_count * numrecordedbands * cfg_numBufferedFFTs, cuStream);

    // Need to make sure that this allocation has completed before we try to access the data
    unpackeddata_gpu->sync();

    for (int j = 0; j < cfg_numBufferedFFTs; j++) {
        for (size_t i = 0; i < numrecordedbands; i++) {
            unpackedarrays_gpu->ptr()[(j * numrecordedbands) + i] =
                    unpackeddata_gpu->ptr() + (((j * numrecordedbands) + i) * unpackedarrays_elem_count);
        }
    }

    estimatedbytes_gpu += unpackedarrays_gpu->size();
    estimatedbytes_gpu += unpackeddata_gpu->size();

    // Copy the unpacked gpu arrays to the device - these won't change again
    unpackedarrays_gpu->copyToDevice();

    gSampleIndexes = new GpuMemHelper<int>(cfg_numBufferedFFTs, cuStream);
    gValidSamples = new GpuMemHelper<bool>(cfg_numBufferedFFTs, cuStream);
    gInterpolator = new GpuMemHelper<double>(interpolator, 3, cuStream);
    gFracSampleError = new GpuMemHelper<float>(cfg_numBufferedFFTs, cuStream);

    gLoFreqs = new GpuMemHelper<double>(numrecordedfreqs, cuStream);

    indices = new GpuMemHelper<int>(10, cuStream);

    // Copy the lofreq values to the GPU
    for (auto i = 0; i < numrecordedfreqs; i++) {
        gLoFreqs->ptr()[i] = config->getDRecordedFreq(configindex, datastreamindex, i);
    }

    gLoFreqs->copyToDevice();

    int n[] = {fftchannels};
    int istride = 1;
    int ostride = 1;
    int idist = fftchannels;
    int odist = fftchannels;

    int inembed[] = {0};
    int onembed[] = {0};

    checkCufft(
            cufftPlanMany(
                    &fft_plan,
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


    // precalc
    nearestSamples = new int[cfg_numBufferedFFTs];

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "GPUMode(): " << duration.count() << endl;

    constructor_time = high_resolution_clock::now();
}

static unsigned long long avg_unpack;
static unsigned long long avg_copyto;
static unsigned long long avg_rotate;
static unsigned long long avg_fft;
static unsigned long long avg_fracrotate;
static unsigned long long avg_postprocess;
static unsigned long long processing_time;

int calls = 0;

GPUMode::~GPUMode() {
    auto start = high_resolution_clock::now();

    delete complexunpacked_gpu;
    delete fftd_gpu;
    delete conj_fftd_gpu;
    delete temp_autocorrelations_gpu;
    delete unpackedarrays_gpu;
    delete unpackeddata_gpu;

    delete gSampleIndexes;
    delete gValidSamples;
    delete gInterpolator;
    delete gFracSampleError;

    delete[] nearestSamples;

    checkCufft(cufftDestroy(fft_plan));
    checkCuda(cudaStreamDestroy(cuStream));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "~GPUMode(): " << duration.count() << endl;

    cout << "Average unpack: " << avg_unpack / calls << endl;
    cout << "Average copyto: " << avg_copyto / calls << endl;
    cout << "Average rotate: " << avg_rotate / calls << endl;
    cout << "Average fft: " << avg_fft / calls << endl;
    cout << "Average fracrotate: " << avg_fracrotate / calls << endl;
    cout << "Average postprocess: " << avg_postprocess / calls << endl;
    cout << "Actual time processing (seconds): " << (double) processing_time / 1000. / 1000. / 3 << endl;

    duration = duration_cast<microseconds>(stop - constructor_time);
    cout << "GPUMode lifetime: " << duration.count() / 1000. / 1000. << endl;
}

int GPUMode::process_gpu(int fftloop, int numBufferedFFTs, int startblock,
                         int numblocks)  //frac sample error is in microseconds
{
    auto begin_time = high_resolution_clock::now();

    calls += 1;
//    std::cout << "Doing the thing. fftloop: " << fftloop << ", numBufferedFFTs: " << numBufferedFFTs << ", numblocks: " << numblocks << ", startblock: " << startblock << std::endl;

    // Sanity checks
    if (!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) {
        NOT_SUPPORTED("DPhaseCal");
    }

    if (fringerotationorder != 1) { // linear only
        NOT_SUPPORTED("fringerotationorder = " + to_string(fringerotationorder));
    }

    if (usedouble) {
        NOT_SUPPORTED("usedouble branch");
    }

    for (auto i = 0; i < numrecordedfreqs; i++) {
        if (recordedfreqlooffsets[i] > 0.0 || recordedfreqlooffsets[i] < 0.0) {
            NOT_SUPPORTED("lo offsets");
        }
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

    auto start = high_resolution_clock::now();

    // Reset the autocorrelations
    checkCuda(cudaMemsetAsync(temp_autocorrelations_gpu->gpuPtr(), 0, sizeof(cf32) * numrecordedbands * recordedbandchannels * 3, cuStream));

    // Update the interpolator
    gInterpolator->copyToDevice();

    calculatePre_cpu(fftloop, numBufferedFFTs, startblock, numblocks);

    // First unpack all the data
    int numfftsprocessed = 0;
    for (; numfftsprocessed < numBufferedFFTs; numfftsprocessed++) {
        int i = fftloop * numBufferedFFTs + numfftsprocessed + startblock;
        if (i >= startblock + numblocks)
            break; // may not have to fully complete last fftloop

        process_unpack(i, numfftsprocessed);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "unpack: " << duration.count() << endl;
    avg_unpack += duration.count();

    start = high_resolution_clock::now();

    // Copy the data to the gpu
    unpackeddata_gpu->copyToDevice();

    // We need to copy the sample indexes to the gpu
    gSampleIndexes->copyToDevice();
    gValidSamples->copyToDevice();

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "copy: " << duration.count() << endl;
    avg_copyto += duration.count();

    start = high_resolution_clock::now();

    // Run the fringe rotation
    fringeRotation(fftloop, numBufferedFFTs, startblock, numblocks);

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "rotate: " << duration.count() << endl;
    avg_rotate += duration.count();

    start = high_resolution_clock::now();
    // Actually run the FFT
    runFFT();

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "fft: " << duration.count() << endl;
    avg_fft += duration.count();

    start = high_resolution_clock::now();

    // do the frac sample correct (+ phase shifting if applicable, + fringe rotate if its post-f)
    fractionalRotation(fftloop, numBufferedFFTs, startblock, numblocks);

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "fracrotate: " << duration.count() << endl;
    avg_fracrotate += duration.count();

    start = high_resolution_clock::now();

    // This synchronise is really needed, as we need the GPU processing/memcpys to finish before we read the result
    // data in to the autocorrelation vectors
    temp_autocorrelations_gpu->sync();

    // Copy over the autocorrs
    for (int j = 0; j < numrecordedbands; j++) {
        vectorCopy_cf32(
                reinterpret_cast<const cf32*>(&temp_autocorrelations_gpu->ptr()[(j * recordedbandchannels * 3)]),
                autocorrelations[0][j],
                recordedbandchannels);
    }

    if (numrecordedbands > 1) {
        //if we need to, do the cross-polar autocorrelations
        vectorCopy_cf32(reinterpret_cast<const cf32*>(&temp_autocorrelations_gpu->ptr()[recordedbandchannels]),
                        autocorrelations[1][0],
                        recordedbandchannels);

        vectorCopy_cf32(reinterpret_cast<const cf32*>(&temp_autocorrelations_gpu->ptr()[recordedbandchannels * 2]),
                        autocorrelations[1][1],
                        recordedbandchannels);
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
    if (nearestSamples[subloopindex] < -1 ||
        (((nearestSamples[subloopindex] + fftchannels) / samplesperblock) * bytesperblocknumerator) / bytesperblockdenominator >
        datalengthbytes) {
//        std::cerr << "to M::p_g; we are in the 'crap data' branch" << std::endl;
//        cerror << startl << "MODE error for datastream " << datastreamindex
//               << " - trying to process data outside range - aborting!!! nearest sample was " << nearestSamples[subloopindex]
//               << ", the max bytes should be " << datalengthbytes << " and hence last sample should be "
//               << (datalengthbytes * bytesperblockdenominator) / (bytesperblocknumerator * samplesperblock)
//               << " (fftchannels is " << fftchannels << "), offsetseconds was " << offsetseconds << ", offsetns was "
//               << offsetns << ", index was " << index << ", average delay was " << nearestSamples[subloopindex] << ", datasec was "
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
    // Clear the perbandweights for this subloopindex
    if(perbandweights)
    {
        for(int b = 0; b < numrecordedbands; ++b)
        {
            perbandweights[subloopindex][b] = 0.0;
        }
    }

    if (!is_data_valid(index, subloopindex)) {
        // since these data weights can be retreived after this processing ends, reset them to a default of zero in case they don't get updated
        dataweight[subloopindex] = 0.0;

        gValidSamples->ptr()[subloopindex] = false;
        return;
    }

    gValidSamples->ptr()[subloopindex] = true;

    if (nearestSamples[subloopindex] == -1) {
        nearestSamples[subloopindex] = 0;
        dataweight[subloopindex] = unpack(nearestSamples[subloopindex], subloopindex);
    } else if (nearestSamples[subloopindex] < unpackstartsamples || nearestSamples[subloopindex] > unpackstartsamples + unpacksamples - fftchannels)
        //need to unpack more data
        dataweight[subloopindex] = unpack(nearestSamples[subloopindex], subloopindex);

    gSampleIndexes->ptr()[subloopindex] = nearestSamples[subloopindex] - unpackstartsamples;

    if (!is_dataweight_valid(subloopindex)) {
        gValidSamples->ptr()[subloopindex] = false;
    } else {
        for (int i = 0; i < numrecordedfreqs; i++) {
            int count = 0;
            // PWCR numrecordedbands = 2 for the test; but e.g. 8 is very realistical
            // Loop over all recorded bands looking for the matching frequency we should be dealing with
            for (int j = 0; j < numrecordedbands; j++) {
                // For upper sideband bands, normally just need to copy the fftd channels.
                // However for complex double upper sideband, the two halves of the frequency space are swapped, so they need to be swapped back

                if (config->matchingRecordedBand(configindex, datastreamindex, i, j)) {
                    indices->ptr()[count++] = j;

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

                    //store the weight for the autocorrelations
                    if(perbandweights)
                    {
                        weights[0][j] += perbandweights[subloopindex][j];
                    }
                    else
                    {
                        weights[0][j] += dataweight[subloopindex];
                    }
                }
            }

            if (count > 1) {
                //store the weights
                if(perbandweights)
                {
                    weights[1][indices->ptr()[0]] += perbandweights[subloopindex][indices->ptr()[0]]*perbandweights[subloopindex][indices->ptr()[1]];
                    weights[1][indices->ptr()[1]] += perbandweights[subloopindex][indices->ptr()[0]]*perbandweights[subloopindex][indices->ptr()[1]];
                }
                else
                {
                    weights[1][indices->ptr()[0]] += dataweight[subloopindex];
                    weights[1][indices->ptr()[1]] += dataweight[subloopindex];
                }
            }
        }
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
        nearestSamples[subloopindex] = int(starttime / sampletime + 0.5);

        double nearestsampletime = nearestSamples[subloopindex] * sampletime;
        gFracSampleError->ptr()[subloopindex] = float(starttime - nearestsampletime);
    }

    // Start copying the fracSampleErrors to the gpu
    gFracSampleError->copyToDevice();
}

__global__ void _gpu_fringeRotation(
        cuFloatComplex* const dest,
        float **const src,
        const double* const interpolator,
        const int* const sampleIndexes,
        const bool* const validSamples,
        const double* const lofreqs,
        double sampletime,
        double recordedfreqlooffset,
        int numrecordedfreqs,
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

    /* The actual calculation that is going on for the linear case is as follows:

     Calculate complexrotator[j]  (for j = 0 to fftchanels-1) as:

     complexrotator[j] = exp( 2 pi i * (A*j + B) )

     where:

     A = a*lofreq/fftchannels - sampletime*1.0e-6*recordedfreqlooffsets[i]
     B = b*lofreq + fraclofreq*integerdelay - recordedfreqlooffsets[i]*fracwalltime - fraclooffset*intwalltime

     And a, b are computed outside the recordedfreq loop (variable i)
    */

    // Calculate littleA/B
    double d0 = interpolator[0] * index * index + interpolator[1] * index + interpolator[2];
    double d1 = interpolator[0] * (index + 0.5) * (index + 0.5) + interpolator[1] * (index + 0.5) + interpolator[2];
    double d2 = interpolator[0] * (index + 1) * (index + 1) + interpolator[1] * (index + 1) + interpolator[2];

    double a = d2 - d0;
    double b = d0 + (d1 - (a * 0.5 + d0)) / 3.0;

    for (size_t i = 0; i < numrecordedfreqs; i++) {
        // Calculate BigA/B
        double bigAval = a * lofreqs[i] / fftchannels - sampletime * 1.e-6 * recordedfreqlooffset;
        double bigBval = b * lofreqs[i];

        // Calculate
        double bigB_reduced = bigBval - int(bigBval);
        double exponent = (bigAval * channelindex + bigB_reduced);
        exponent -= int(exponent);
        cuFloatComplex cr;
        sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
        cuFloatComplex c = make_cuFloatComplex(srcVal, 0.f);
        dest[destIndex] = cuCmulf(c, cr);
    }
}

void GPUMode::fringeRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {

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

    _gpu_fringeRotation<<<
        dim3(numBufferedFFTs, fftchannels_grid),
        dim3(numrecordedbands,fftchannels_block),
        0, cuStream
    >>>
            (
                    complexunpacked_gpu->gpuPtr(),
                    unpackedarrays_gpu->gpuPtr(),
                    gInterpolator->gpuPtr(),
                    gSampleIndexes->gpuPtr(),
                    gValidSamples->gpuPtr(),
                    gLoFreqs->gpuPtr(),
                    sampletime,
                    recordedfreqlooffsets[0],
                    numrecordedfreqs,
                    fftloop,
                    startblock,
                    numblocks,
                    fftchannels
            );
}

// Adapted from https://forums.developer.nvidia.com/t/atomic-add-for-complex-numbers/39757
__device__ void atomicAddFloatComplex(cuFloatComplex* a, cuFloatComplex b){
    // transform the addresses of real and imag. parts to double pointers
    float *x = (float*)a;
    float *y = x+1;
    //use atomicAdd for double variables
    atomicAdd(x, cuCrealf(b));
    atomicAdd(y, cuCimagf(b));
}

__global__ void _gpu_resultsrotatorMultiply(
        cuFloatComplex* const fftoutputs,
        cuFloatComplex* const conjfftoutputs,
        cuFloatComplex* const autocorrelations,
        const float* const fracSampleError,
        const bool* const validSamples,
        const double recordedbandwidth,
        double recordedfreqclockoffset,
        double recordedfreqclockoffsetdelta,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels,
        size_t recordedbandchannels,
        size_t numrecordedbands
    ) {
    // numBufferedFFTs(blockIdx.x) * fftchannels(threadIdx.x)

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the fftchannels_block size
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

    const size_t channelindex = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (channelindex >= recordedbandchannels) {
        return;
    }

    for (size_t bandindex = 0; bandindex < numrecordedbands; bandindex++) {
        // Calculate the destination index
        const size_t dataIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;
        const size_t autocorrIndex = (bandindex * recordedbandchannels * 3) + channelindex;


        /* Creating a fractional sample rotation array
         *  The actual calculation being performed is as follows:
         *  Assume we know the frequency of every FFT output channel, and it is stored in an array of length fftchannels, called channelfreq
         *  then for every frequency subband f (in the range 0 … recordedbandchannels), calculate the slope as:
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

        // Calculate fracsampleerror - recordedfreqclockoffsets[f] + recordedfreqclockoffsetsdelta[f]/2
        double bigAval = fracSampleError[subloopindex] - recordedfreqclockoffset + recordedfreqclockoffsetdelta / 2;

        // Calculate fftchannels[j] = recordedbandwidth * j / fftchannels
        double subFreq = recordedbandwidth * channelindex / recordedbandchannels;

        // Calculate
        double exponent = bigAval * subFreq;
        exponent -= int(exponent);
        cuFloatComplex cr;
        sincosf(TWO_PI * exponent, &cr.y, &cr.x);
        fftoutputs[dataIndex] = cuCmulf(fftoutputs[dataIndex], cr);

        // do the conjugation
        conjfftoutputs[dataIndex] = cuConjf(fftoutputs[dataIndex]);

        // do the autocorrelation (skipping Nyquist channel)
        atomicAddFloatComplex(&autocorrelations[autocorrIndex], cuCmulf(fftoutputs[dataIndex], conjfftoutputs[dataIndex]));
    }

    if (numrecordedbands > 1) {
        // if we need to, do the cross-polar autocorrelations
        size_t fftIndex = (subloopindex * fftchannels * numrecordedbands) + (0 * fftchannels) + channelindex;
        size_t conjIndex = (subloopindex * fftchannels * numrecordedbands) + (1 * fftchannels) + channelindex;

        atomicAddFloatComplex(&autocorrelations[recordedbandchannels + channelindex], cuCmulf(fftoutputs[fftIndex], conjfftoutputs[conjIndex]));

        fftIndex = (subloopindex * fftchannels * numrecordedbands) + (1 * fftchannels) + channelindex;
        conjIndex = (subloopindex * fftchannels * numrecordedbands) + (0 * fftchannels) + channelindex;

        atomicAddFloatComplex(&autocorrelations[recordedbandchannels * 2 + channelindex], cuCmulf(fftoutputs[fftIndex], conjfftoutputs[conjIndex]));
    }
}

void GPUMode::fractionalRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {
    // At this point we have
    // * FFT results on GPU
    // * subchannelfreqs
    // * Which samples are valid - ie that we need to operate on

    // numBufferedFFTs(blockIdx.x) * fftchannels(threadIdx.x)
    size_t fftchannels_block;
    size_t fftchannels_grid;

    size_t divisor = cudaMaxThreadsPerBlock;
    if (recordedbandchannels > divisor) {
        fftchannels_block = divisor;
        fftchannels_grid = recordedbandchannels / divisor;

        if (recordedbandchannels % divisor != 0) {
            fftchannels_grid++;
        }
    } else {
        fftchannels_block = recordedbandchannels;
        fftchannels_grid = 1;
    }

    _gpu_resultsrotatorMultiply<<<dim3(numBufferedFFTs, fftchannels_grid), dim3(fftchannels_block), 0, cuStream>>>
            (
                    fftd_gpu->gpuPtr(),
                    conj_fftd_gpu->gpuPtr(),
                    temp_autocorrelations_gpu->gpuPtr(),
                    gFracSampleError->gpuPtr(),
                    gValidSamples->gpuPtr(),
                    recordedbandwidth,
                    recordedfreqclockoffsets[0],
                    recordedfreqclockoffsetsdelta[0],
                    fftloop,
                    startblock,
                    numblocks,
                    fftchannels,
                    recordedbandchannels,
                    numrecordedbands
            );

    // Start copying the autocorrelations back to the host
    temp_autocorrelations_gpu->copyToHost();
}

void GPUMode::runFFT() {
    checkCufft(cufftExecC2C(fft_plan, complexunpacked_gpu->gpuPtr(), fftd_gpu->gpuPtr(), CUFFT_FORWARD));
}
