#define NOT_SUPPORTED(x) std::cerr << "Whoops, we don't support this on the GPU: " << x << std::endl; exit(1)

#include "gpumode.h"
#include "alert.h"
#include <cuda_runtime.h>
#include <string>

#include "gpumode_kernels.cuh"

// Host calls

GPUMode::GPUMode(Configuration * conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend, int gsamples, int nrecordedfreqs, double recordedbw, double * recordedfreqclkoffs, double * recordedfreqclkoffsdelta, double * recordedfreqphaseoffs, double * recordedfreqlooffs, int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling, Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular, int fringerotorder, int arraystridelen, bool cacorrs, double bclock):
    Mode(conf, confindex, dsindex, recordedbandchan, chanstoavg, bpersend, gsamples, nrecordedfreqs, recordedbw, recordedfreqclkoffs, recordedfreqclkoffsdelta, recordedfreqphaseoffs, recordedfreqlooffs, nrecordedbands, nzoombands, nbits, sampling, tcomplex, unpacksamp, fbank, linear2circular, fringerotorder, arraystridelen, cacorrs, bclock),
    estimatedbytes_gpu(0)
  {
  checkCuda(cudaMalloc(&this->subxoff_gpu,
        sizeof(double)*this->arraystridelength));
  checkCuda(cudaMemcpy(this->subxoff_gpu, this->subxoff,
        sizeof(double)*this->arraystridelength, cudaMemcpyHostToDevice));
  checkCuda(cudaMalloc(&this->subxval_gpu,
        sizeof(double)*this->arraystridelength));

  this->complexunpacked_gpu = gpu_malloc<cuFloatComplex>(this->fftchannels);
  this->estimatedbytes_gpu += sizeof(cuFloatComplex)*this->fftchannels;

  this->fftd_gpu = gpu_malloc<cuFloatComplex>(this->fftchannels);
  this->estimatedbytes_gpu += sizeof(cuFloatComplex)*this->fftchannels;

  this->complexrotator_gpu = gpu_malloc<cuFloatComplex>(this->fftchannels);

  this->unpackedarrays_gpu = new float*[numrecordedbands];
  this->estimatedbytes += sizeof(float*)*numrecordedbands;
  for(size_t i = 0; i < numrecordedbands; ++i) {
    checkCuda(cudaMalloc(&this->unpackedarrays_gpu[i], sizeof(float)*unpacksamples));
    this->estimatedbytes_gpu += sizeof(float)*this->unpacksamples;
  }
  // TODO: PWC: allocations for complex

  checkCufft(cufftPlan1d(&this->fft_plan, this->fftchannels, CUFFT_C2C, 1));
}

GPUMode::~GPUMode() {
  checkCuda(cudaFree(this->complexunpacked_gpu));
  checkCuda(cudaFree(this->fftd_gpu));

  for(size_t i = 0; i < numrecordedbands; ++i) {
    checkCuda(cudaFree(this->unpackedarrays_gpu[i]));
  }
  delete [] this->unpackedarrays_gpu;
  // TODO: PWC: dealloctions for complex

  cufftDestroy(this->fft_plan);
}

// typical numrecordedbands = 2

void GPUMode::process(int index, int subloopindex)  //frac sample error is in microseconds 
{
#ifndef NEUTERED_DIFX
  static int nth_call = 0;
  ++nth_call;
  double phaserotation, averagedelay, nearestsampletime, starttime, lofreq, walltimesecs, fracwalltime, fftcentre, d0, d1, d2, fraclooffset;
  f32 phaserotationfloat, fracsampleerror;
  int status, count, nearestsample, integerdelay, RcpIndex, LcpIndex, intwalltime;
  cf32* fftptr;
  f32* currentstepchannelfreqs;
  f32* currentsubchannelfreqs;
  int indices[10];
  bool looff, isfraclooffset;
  //cout << "For Mode of datastream " << datastreamindex << ", index " << index << ", validflags is " << validflags[index/FLAGS_PER_INT] << ", after shift you get " << ((validflags[index/FLAGS_PER_INT] >> (index%FLAGS_PER_INT)) & 0x01) << endl;

  //since these data weights can be retreived after this processing ends, reset them to a default of zero in case they don't get updated
  dataweight[subloopindex] = 0.0;
  if(perbandweights)
  {
    NOT_SUPPORTED("per band weights");
  }
  
  if((datalengthbytes <= 1) || (offsetseconds == INVALID_SUBINT) || (((validflags[index/FLAGS_PER_INT] >> (index%FLAGS_PER_INT)) & 0x01) == 0))
  {
    //std::cout << "call " << nth_call << "to M::p_g; we are in the weird place with the datalengthbytes" << std::endl;
    //std::cout << "call " << nth_call << "to M::p_g; numrecorededbands = " << numrecordedbands << std::endl;
    for(int i=0;i<numrecordedbands;i++)
    {
      status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
      if(status != vecNoErr)
        csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
      status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
      if(status != vecNoErr)
        csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
    }
    //cout << "Mode for DS " << datastreamindex << " is bailing out of index " << index << "/" << subloopindex << " which is scan " << currentscan << ", sec " << offsetseconds << ", ns " << offsetns << " because datalengthbytes is " << datalengthbytes << " and validflag was " << ((validflags[index/FLAGS_PER_INT] >> (index%FLAGS_PER_INT)) & 0x01) << endl;
    return; //don't process crap data
  }

  fftcentre = index+0.5;
  averagedelay = interpolator[0]*fftcentre*fftcentre + interpolator[1]*fftcentre + interpolator[2];
  fftstartmicrosec = index*fftchannels*sampletime; //CHRIS CHECK
  starttime = (offsetseconds-datasec)*1000000.0 + (static_cast<long long>(offsetns) - static_cast<long long>(datans))/1000.0 + fftstartmicrosec - averagedelay;
  nearestsample = int(starttime/sampletime + 0.5);
  walltimesecs = model->getScanStartSec(currentscan, config->getStartMJD(), config->getStartSeconds()) + offsetseconds + offsetns/1.0e9 + fftstartmicrosec/1.0e6;
  intwalltime = static_cast<int>(walltimesecs);
  fracwalltime = walltimesecs - intwalltime;
  //cinfo << startl << "ATD: fftstartmicrosec " << fftstartmicrosec << ", sampletime " << sampletime << ", fftchannels " << fftchannels << ", bytesperblocknumerator " << bytesperblocknumerator << ", nearestsample " << nearestsample << endl;

  //if we need to, unpack some more data - first check to make sure the pos is valid at all
  //cout << "Datalengthbytes for " << datastreamindex << " is " << datalengthbytes << endl;
  //cout << "Fftchannels for " << datastreamindex << " is " << fftchannels << endl;
  //cout << "samplesperblock for " << datastreamindex << " is " << samplesperblock << endl;
  //cout << "nearestsample for " << datastreamindex << " is " << nearestsample << endl;
  //cout << "bytesperblocknumerator for " << datastreamindex << " is " << bytesperblocknumerator << endl;
  //cout << "bytesperblockdenominator for " << datastreamindex << " is " << bytesperblockdenominator << endl;
  if(nearestsample < -1 || (((nearestsample + fftchannels)/samplesperblock)*bytesperblocknumerator)/bytesperblockdenominator > datalengthbytes)
  {
    std::cout << "call " << nth_call << "to M::p_g; we are in the 'crap data' branch" << std::endl;
    cerror << startl << "MODE error for datastream " << datastreamindex << " - trying to process data outside range - aborting!!! nearest sample was " << nearestsample << ", the max bytes should be " << datalengthbytes << " and hence last sample should be " << (datalengthbytes*bytesperblockdenominator)/(bytesperblocknumerator*samplesperblock)  << " (fftchannels is " << fftchannels << "), offsetseconds was " << offsetseconds << ", offsetns was " << offsetns << ", index was " << index << ", average delay was " << averagedelay << ", datasec was " << datasec << ", datans was " << datans << ", fftstartmicrosec was " << fftstartmicrosec << endl;
    for(int i=0;i<numrecordedbands;i++)
    {
      status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
      if(status != vecNoErr)
        csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
      status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
      if(status != vecNoErr)
        csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
    }
    return;
  }
  if(nearestsample == -1)
  {
    nearestsample = 0;
    dataweight[subloopindex] = unpack(nearestsample, subloopindex);
  }
  else if(nearestsample < unpackstartsamples || nearestsample > unpackstartsamples + unpacksamples - fftchannels)
    //need to unpack more data
    dataweight[subloopindex] = unpack(nearestsample, subloopindex);

 /*
  * After DiFX-2.4, it is proposed to change the handling of lower sideband and dual sideband data, such
  * that the data are manipulated here (directly after unpacking) to ensure that it appears like single 
  * sideband USB data.  In order to do that, we will loop over all recorded bands, performing the 
  * following checks and if necessary manipulations:
  * 1) if band sense is LSB, cast the unpacked data as a complex f32 and conjugate it.  If it was a complex
  *    sampled band, this flips the sideband.  If it was a real sampled band, then every second sample
  *    will be multiplied by -1, which is exactly was is required to flip the sideband also.
  *    *** NOTE: For real data, will need to use fracwalltimesecs plus the sampling rate to determine
  *              whether it is necessary to offset the start of the vector by one sample.
  * 2) Now the frequencies definitely run from most negative to most positive, but we also want the lowest
  *    frequency channel to be "DC", and this is not the case for complex double sideband data.  So for
  *    complex double sideband data, rotate the unpacked data by e^{-i 2 pi BW t} to shift the most negative
  *    frequency component up to 0 Hz.  Need to use wallclocksecs for time here too.
  * Now nothing in mode.cpp or core.cpp needs to know about whether the data was originally lower sideband
  * or not.  That will mean taking out some of the current logic, pretty much all to do with fractional sample
  * correction.
  * 
  * Some other specific implementation notes:
  * - Need to do this straight after an unpack, for the whole unpacksamples, so the two calls to unpack()
  *   above will need to be combined.
  * - It may be profitable to move the LO offset correction up to here also, and possibly also to refactor 
  *   it to change the steptval array rather than doing a separate addition. (although a separate addition
  *   for fraclooffset if required would still be needed).  Be careful of zero-order fringe rotation.
  * - lsbfracsample arrays will need to be removed, as will the checks that select them.
  * - Elsewhere, it will probably be preferable to maintain information slightly differently (for each
  *   subband, maintain lower edge frequency, bandwidth, SSLO, sampling type [real/complex], matching band).
  *   This would be in configuration.cpp/.h, maybe also vex2difx?
  * - mark5access has option to unpack real data as complex - could consider using this to save time.  
  *   Would need to make a similar option for LBA data.
  */

  if(!(dataweight[subloopindex] > 0.0)) {
    for(int i=0;i<numrecordedbands;i++)
    {
      status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
      if(status != vecNoErr)
        csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
      status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
      if(status != vecNoErr)
        csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
    }
    return;
  }

  nearestsampletime = nearestsample*sampletime;
  fracsampleerror = float(starttime - nearestsampletime);

  if(!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) 
  {
    NOT_SUPPORTED("DPhaseCal");
  }

  integerdelay = 0;
  //std::cout << "call " << nth_call << "to M::p_g; fringerotationorder = " << fringerotationorder << std::endl;
  switch(fringerotationorder) {
    case 0: //post-F
      NOT_SUPPORTED("fringerotationorder = 1");
      break;
    case 1: //linear
      d0 = interpolator[0]*index*index + interpolator[1]*index + interpolator[2];
      d1 = interpolator[0]*(index+0.5)*(index+0.5) + interpolator[1]*(index+0.5) + interpolator[2];
      d2 = interpolator[0]*(index+1)*(index+1) + interpolator[1]*(index+1) + interpolator[2];
      a = d2-d0;
      b = d0 + (d1 - (a*0.5 + d0))/3.0;
      integerdelay = static_cast<int>(b);
      b -= integerdelay;

      status = vectorMulC_f64(subxoff, a, subxval, arraystridelength);
      if(status != vecNoErr)
        csevere << startl << "Error in linearinterpolate, subval multiplication" << endl;
      status = vectorMulC_f64(stepxoff, a, stepxval, numfrstrides);
      if(status != vecNoErr)
        csevere << startl << "Error in linearinterpolate, stepval multiplication" << endl;
      status = vectorAddC_f64_I(b, subxval, arraystridelength);
      if(status != vecNoErr)
        csevere << startl << "Error in linearinterpolate, subval addition!!!" << endl;
      break;
    case 2: //quadratic
      NOT_SUPPORTED("fringerotationorder = 2");
      break;
    default: //shouldn't happen
      NOT_SUPPORTED("unrecognised fringe rotation order");
      break;
  }

  // Do the main work here
  // Loop over each frequency and to the fringe rotation and FFT of the data

  if(1 != numrecordedfreqs)
  {
    NOT_SUPPORTED("a value for 'numrecordedfreqs' other than 1");
  }
  const int i = 0; // Was the 'numrecordedfreqs' loop index
  count = 0;
  //updated so that Nyquist channel is not accumulated for either USB or LSB data
  //and is excised entirely, so both USB and LSB data start at the same place (no sidebandoffset)
  currentstepchannelfreqs = stepchannelfreqs;
  currentsubchannelfreqs = subchannelfreqs;
  if(usedouble)
  {
    NOT_SUPPORTED("usedouble branch");
  }
  else
  {
    if(config->getDRecordedLowerSideband(configindex, datastreamindex, i))
    {
      currentstepchannelfreqs = lsbstepchannelfreqs;
    }
  }

  looff = false;
  isfraclooffset = false;
  if(recordedfreqlooffsets[i] > 0.0 || recordedfreqlooffsets[i] < 0.0) {
    NOT_SUPPORTED("lo offsets");
  }

  //get ready to apply fringe rotation, if it is pre-F.  
  //By default, the local oscillator frequency (which is used for fringe rotation) is the band edge, as specified inthe input file
  lofreq = config->getDRecordedFreq(configindex, datastreamindex, i);

  // For double-sideband data, the LO frequency is at the centre of the band, not the band edge
  if (usecomplex && usedouble)
  {
    NOT_SUPPORTED("complex double-sideband data");
  } else if(usecomplex) {
    NOT_SUPPORTED("complex data");
  }
  //std::cout << "lo freq: " << lofreq << std::endl;

  // OK, now let's put some actual GPU in here
  switch(fringerotationorder) {
    case 1: // linear

/* The actual calculation that is going on for the linear case is as follows:

 Calculate complexrotator[j]  (for j = 0 to fftchanels-1) as:

 complexrotator[j] = exp( 2 pi i * (A*j + B) )

 where:

 A = a*lofreq/fftchannels - sampletime*1.0e-6*recordedfreqlooffsets[i]
 B = b*lofreq/fftchannels + fraclofreq*integerdelay - recordedfreqlooffsets[i]*fracwalltime - fraclooffset*intwalltime

 And a, b are computed outside the recordedfreq loop (variable i)
*/

      status = vectorMulC_f64(subxval, lofreq, subphase, arraystridelength);
      if(status != vecNoErr)
        csevere << startl << "Error in linearinterpolate lofreq sub multiplication!!!" << status << endl;
      status = vectorMulC_f64(stepxval, lofreq, stepphase, numfrstrides);
      if(status != vecNoErr)
        csevere << startl << "Error in linearinterpolate lofreq step multiplication!!!" << status << endl;
      if(fractionalLoFreq) {
        status = vectorAddC_f64_I((lofreq-int(lofreq))*double(integerdelay), subphase, arraystridelength);
        if(status != vecNoErr)
          csevere << startl << "Error in linearinterpolate lofreq non-integer freq addition!!!" << status << endl;
      }
      // if(looff) { -- PWC removed this if() ... }
      if(looff) {
        NOT_SUPPORTED("looff");
      }
      for(int j=0;j<arraystridelength;j++) { // PWCR - typ 16
        subarg[j] = -TWO_PI*(subphase[j] - int(subphase[j]));
      }
      for(int j=0;j<numfrstrides;j++) { // PWCR - typ 16
        steparg[j] = -TWO_PI*(stepphase[j] - int(stepphase[j]));
      }
      status = vectorSinCos_f32(subarg, subsin, subcos, arraystridelength);
      if(status != vecNoErr)
        csevere << startl << "Error in sin/cos of sub rotate argument!!!" << endl;
      status = vectorSinCos_f32(steparg, stepsin, stepcos, numfrstrides);
      if(status != vecNoErr)
        csevere << startl << "Error in sin/cos of step rotate argument!!!" << endl;
      status = vectorRealToComplex_f32(subcos, subsin, complexrotator, arraystridelength);
      if(status != vecNoErr)
        csevere << startl << "Error assembling sub into complex!!!" << endl;
      status = vectorRealToComplex_f32(stepcos, stepsin, stepcplx, numfrstrides);
      if(status != vecNoErr)
        csevere << startl << "Error assembling step into complex!!!" << endl;
      for(int j=1;j<numfrstrides;j++) {
        status = vectorMulC_cf32(complexrotator, stepcplx[j], &complexrotator[j*arraystridelength], arraystridelength);
        if(status != vecNoErr)
          csevere << startl << "Error doing the time-saving complex multiplication!!!" << endl;
      }
      break;
    case 2: // Quadratic
      NOT_SUPPORTED("fringerotationorder == 2");
      break;
    default: // Quadratic
      NOT_SUPPORTED("fringerotationorder != 1");
      break;
  }

  // Note recordedfreqclockoffsetsdata will usually be zero, but avoiding if statement
  status = vectorMulC_f32(currentsubchannelfreqs, fracsampleerror - recordedfreqclockoffsets[i] + recordedfreqclockoffsetsdelta[i]/2, subfracsamparg, arraystridelength);
  if(status != vecNoErr) {
    csevere << startl << "Error in frac sample correction, arg generation (sub)!!!" << status << endl;
    exit(1);
  }
  status = vectorMulC_f32(currentstepchannelfreqs, fracsampleerror - recordedfreqclockoffsets[i] + recordedfreqclockoffsetsdelta[i]/2, stepfracsamparg, numfracstrides/2);
  if(status != vecNoErr)
    csevere << startl << "Error in frac sample correction, arg generation (step)!!!" << status << endl;

  // For zero-th order (post-F) fringe rotation, calculate the fringe rotation (+ LO offset if necessary)
  if(fringerotationorder == 0) { // do both LO offset and fringe rotation  (post-F)
    NOT_SUPPORTED("fringerotationorder == 0");
  }

  //create the fractional sample correction array
  status = vectorSinCos_f32(subfracsamparg, subfracsampsin, subfracsampcos, arraystridelength);
  if(status != vecNoErr)
    csevere << startl << "Error in frac sample correction, sin/cos (sub)!!!" << status << endl;
  status = vectorSinCos_f32(stepfracsamparg, stepfracsampsin, stepfracsampcos, numfracstrides/2);
  if(status != vecNoErr)
    csevere << startl << "Error in frac sample correction, sin/cos (sub)!!!" << status << endl;
  status = vectorRealToComplex_f32(subfracsampcos, subfracsampsin, fracsamprotatorA, arraystridelength);
  if(status != vecNoErr)
    csevere << startl << "Error in frac sample correction, real to complex (sub)!!!" << status << endl;
  status = vectorRealToComplex_f32(stepfracsampcos, stepfracsampsin, stepfracsampcplx, numfracstrides/2);
  if(status != vecNoErr)
    csevere << startl << "Error in frac sample correction, real to complex (step)!!!" << status << endl;
  for(int j=1;j<numfracstrides/2;j++) {
    status = vectorMulC_cf32(fracsamprotatorA, stepfracsampcplx[j], &(fracsamprotatorA[j*arraystridelength]), arraystridelength);
    if(status != vecNoErr)
      csevere << startl << "Error doing the time-saving complex multiplication in frac sample correction!!!" << endl;
  }

  // now do the first arraystridelength elements (which are different from fracsampptr1 for LSB case)
  status = vectorMulC_cf32_I(stepfracsampcplx[0], fracsamprotatorA, arraystridelength);
  if(status != vecNoErr)
  csevere << startl << "Error doing the first bit of the time-saving complex multiplication in frac sample correction!!!" << endl;

  // Repeat the post F correction steps if each pol is different
  if (deltapoloffsets) {
    NOT_SUPPORTED("deltapoloffsets");
  }

  // PWCR numrecordedbands = 2 for the test; but e.g. 8 is very realistical
  // PWCR This is the big and nasty loop that does all the work
  for(int j=0;j<numrecordedbands;j++)  // Loop over all recorded bands looking for the matching frequency we should be dealing with
  {
    if(config->matchingRecordedBand(configindex, datastreamindex, i, j))
    {
      indices[count++] = j;
      switch(fringerotationorder) {
        case 0: //post-F
          NOT_SUPPORTED("fringe rotation order == 0");
          break;
        case 1: // Linear
        case 2: // Quadratic
          if (usecomplex) {
            NOT_SUPPORTED("complex");
          } else {
            // PWCR DO THIS
            status = vectorRealToComplex_f32(&(unpackedarrays[j][nearestsample - unpackstartsamples]), NULL, complexunpacked, fftchannels);
            // TODO: none of this complexrotator faff, just calculate e^...
            // directly in the kernel
            const double bigA_d = a * lofreq/fftchannels - sampletime*1.e-6*recordedfreqlooffsets[i];
            const double bigB_d = b*lofreq   // NOTE - no division by /fftchannels here
                                 + (lofreq-int(lofreq))*integerdelay
                                 - recordedfreqlooffsets[i]*fracwalltime
                                 - fraclooffset*intwalltime;
            for(size_t k = 0; k < fftchannels; ++k) {
              const double exponent_d = (bigA_d * k + bigB_d);
              const std::complex<double> cr_d = exp( std::complex<double>(0, -TWO_PI) * ( exponent_d - int(exponent_d) ) );
              const std::complex<double> cr_orig(this->complexrotator[k].re, this->complexrotator[k].im);
              if(std::abs(cr_d - cr_orig) > 1e-2) {
                std::cerr << " UH OK k = " << k << " and we don't have a match (" << std::abs(cr_d - cr_orig) << ")" << std::endl;
                std::cerr << "       --- (" << cr_d.real() << ", " << cr_d.imag() << ") vs (" << cr_orig.real() << ", " << cr_orig.imag() << ")" << std::endl;
              }
            }
            checkCuda(cudaMemcpy(this->complexrotator_gpu, this->complexrotator, sizeof(cuFloatComplex)*fftchannels, cudaMemcpyHostToDevice));
            gpu_host2DevRtoC(complexunpacked_gpu, &(unpackedarrays[j][nearestsample - unpackstartsamples]), fftchannels);
            //checkCuda(cudaMemcpy(this->complexunpacked_gpu, this->complexunpacked, sizeof(cuFloatComplex)*fftchannels, cudaMemcpyHostToDevice));
            //gpu_inPlaceMultiply_cf(complexrotator_gpu, complexunpacked_gpu, fftchannels);
            gpu_complexrotatorMultiply(this->fftchannels, this->complexunpacked_gpu, bigA_d, bigB_d, this->complexrotator_gpu);
            //sleep(5);
            //exit(42);
            //status = vectorRealToComplex_f32(NULL, NULL, complexunpacked, fftchannels);
            //if (status != vecNoErr)
            //  csevere << startl << "Error in real->complex conversion" << endl;
            status = vectorMul_cf32_I(complexrotator, complexunpacked, fftchannels);
            //if(status != vecNoErr)
            //  csevere << startl << "Error in fringe rotation!!!" << status << endl;
          }
          if(isfft) {
            checkCufft(cufftExecC2C(this->fft_plan, complexunpacked_gpu, fftd_gpu, CUFFT_FORWARD));
              vecFFTSpecC_cf32 * pFFTSpecC;
              status = vectorInitFFTC_cf32(&pFFTSpecC, order, flag, hint, &fftbuffersize, &fftbuffer);
            //status = vectorFFT_CtoC_cf32(complexunpacked, fftd, pFFTSpecC, fftbuffer);
              vectorFreeFFTC_cf32(pFFTSpecC);
            checkCuda(cudaMemcpy(this->fftd, this->fftd_gpu, sizeof(cuFloatComplex)*this->fftchannels, cudaMemcpyDeviceToHost));
          } else {
            NOT_SUPPORTED("!isfft");
          }

          if(config->getDRecordedLowerSideband(configindex, datastreamindex, i)) {
            NOT_SUPPORTED("lower sideband");
          } else {
            // For upper sideband bands, normally just need to copy the fftd channels.
            // However for complex double upper sideband, the two halves of the frequency space are swapped, so they need to be swapped back
            if (usecomplex && usedouble) {
              NOT_SUPPORTED("use complex && usedouble");
            } else {
              status = vectorCopy_cf32(fftd, fftoutputs[j][subloopindex], recordedbandchannels);
            }
          }
          if(status != vecNoErr)
            csevere << startl << "Error copying FFT results!!!" << endl;
          break;
      }

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

      if(dumpkurtosis) //do the necessary accumulation
      {
        NOT_SUPPORTED("dump_kurtosis branch");
      }

      //do the frac sample correct (+ phase shifting if applicable, + fringe rotate if its post-f)
  if (deltapoloffsets==false || config->getDRecordedBandPol(configindex, datastreamindex, j)=='R') {
    status = vectorMul_cf32_I(fracsamprotatorA, fftoutputs[j][subloopindex], recordedbandchannels);
  } else {
    status = vectorMul_cf32_I(fracsamprotatorB, fftoutputs[j][subloopindex], recordedbandchannels);
  }


  if(status != vecNoErr)
    csevere << startl << "Error in application of frac sample correction!!!" << status << endl;

      //do the conjugation
      status = vectorConj_cf32(fftoutputs[j][subloopindex], conjfftoutputs[j][subloopindex], recordedbandchannels);
      if(status != vecNoErr)
        csevere << startl << "Error in conjugate!!!" << status << endl;

  if (!linear2circular) {
    //do the autocorrelation (skipping Nyquist channel)
    status = vectorAddProduct_cf32(fftoutputs[j][subloopindex], conjfftoutputs[j][subloopindex], autocorrelations[0][j], recordedbandchannels);
    if(status != vecNoErr)
      csevere << startl << "Error in autocorrelation!!!" << status << endl;

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
  }


  if (count>1) {
    // Do linear to circular conversion if required
    if (linear2circular) {
      NOT_SUPPORTED("linear to circular polarisation conversion");
    } else if (phasepoloffset) {
      NOT_SUPPORTED("phase polarisation offset");
    }

    //if we need to, do the cross-polar autocorrelations
    if(calccrosspolautocorrs) {
  status = vectorAddProduct_cf32(fftoutputs[indices[0]][subloopindex], conjfftoutputs[indices[1]][subloopindex], autocorrelations[1][indices[0]], recordedbandchannels);
  if(status != vecNoErr)
    csevere << startl << "Error in cross-polar autocorrelation!!!" << status << endl;
  status = vectorAddProduct_cf32(fftoutputs[indices[1]][subloopindex], conjfftoutputs[indices[0]][subloopindex], autocorrelations[1][indices[1]], recordedbandchannels);
  if(status != vecNoErr)
    csevere << startl << "Error in cross-polar autocorrelation!!!" << status << endl;
    
  //store the weights
      if(perbandweights)
      {
    weights[1][indices[0]] += perbandweights[subloopindex][indices[0]]*perbandweights[subloopindex][indices[1]];
    weights[1][indices[1]] += perbandweights[subloopindex][indices[0]]*perbandweights[subloopindex][indices[1]];
      }
      else
      {
    weights[1][indices[0]] += dataweight[subloopindex];
    weights[1][indices[1]] += dataweight[subloopindex];
      }
    }
  }
  
  if (linear2circular) {// Delay this as it is possible for linear2circular to be active, but just one pol present
    NOT_SUPPORTED("linear to circular polarisation conversion");
  }
#endif
}

// vim: shiftwidth=2:softtabstop=2:expandtab
