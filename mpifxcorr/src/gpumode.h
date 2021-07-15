#ifndef GPUMODE_H
#define GPUMODE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include "mode.h"

class Configuration;

class GPUMode: public Mode {
  public:
  GPUMode(Configuration * conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend, int gsamples, int nrecordedfreqs, double recordedbw, double * recordedfreqclkoffs, double * recordedfreqclkoffsdelta, double * recordedfreqphaseoffs, double * recordedfreqlooffs, int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling, Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular, int fringerotorder, int arraystridelen, bool cacorrs, double bclock);
  virtual ~GPUMode();

  void process(int index, int subloopindex);  //frac sample error is in microseconds 

protected:
  double *subxoff_gpu;
  double *subxval_gpu;

  float ** unpackedarrays_gpu;
  cuFloatComplex ** unpackedcomplexarrays_gpu;

  cuFloatComplex *complexunpacked_gpu;

  cuFloatComplex *fftd_gpu;

  size_t estimatedbytes_gpu;

  cuFloatComplex *complexrotator_gpu;

  // Remember how long the 'unpackedarrays' are -- norally this would be
  // 'unpacksamples' but e.g. the Mk5Mode implementation overwrites that
  size_t unpackedarrays_elem_count;
private:

  cufftHandle fft_plan;
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
