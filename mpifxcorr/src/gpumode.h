#ifndef GPUMODE_H
#define GPUMODE_H

#include "mode.h"

class Configuration;

class GPUMode: public Mode {
  public:
  GPUMode(Configuration * conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend, int gsamples, int nrecordedfreqs, double recordedbw, double * recordedfreqclkoffs, double * recordedfreqclkoffsdelta, double * recordedfreqphaseoffs, double * recordedfreqlooffs, int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling, Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular, int fringerotorder, int arraystridelen, bool cacorrs, double bclock);

  void process(int index, int subloopindex);  //frac sample error is in microseconds 

protected:
  double *subxoff_gpu;
  double *subxval_gpu;

};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
