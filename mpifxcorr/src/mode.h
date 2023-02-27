/***************************************************************************
 *   Copyright (C) 2006-2017 by Adam Deller                                *
 *                                                                         *
 *   This program is free software: you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation, either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 ***************************************************************************/
//===========================================================================
// SVN properties (DO NOT CHANGE)
//
// $Id: mode.h 10579 2022-08-02 10:58:00Z JanWagner $
// $HeadURL: https://svn.atnf.csiro.au/difx/mpifxcorr/trunk/src/mode.h $
// $LastChangedRevision: 10579 $
// $Author: JanWagner $
// $LastChangedDate: 2022-08-02 20:58:00 +1000 (Tue, 02 Aug 2022) $
//
//============================================================================
#ifndef MODE_H
#define MODE_H

#include "architecture.h"
#include "configuration.h"
#include "pcal.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuComplex.h>

using namespace std;

/**
@class Mode 
@brief Abstract superclass for all modes.  Provides station-based processing functionality

Possesses all methods necessary for mode functionality but should not be instantiated as it does not build a
lookup table or handle unpacking - this is particular to each subclass of mode.  Station-based processing
(FFT, fringe rotation, fraction sample correction etc) is handled via the process method, based on
the provided data and control (delay) arrays

@author Adam Deller
*/
class Mode{
public:
  f32 **  unpackedarrays;
  cf32 **  unpackedcomplexarrays;
 /**
  * Stores the FFT valid flags for this block of data
  * @param v The array of valid flags for each FFT
  */
  void setValidFlags(s32 * v);

 /**
  * Stores the raw data for the current block series
  * @param d The data array
  * @param dbytes The number of bytes in the data array
  * @param dscan The scan from which the data comes
  * @param dsec The seconds offset from the start of the scan
  * @param dns The offset in nanoseconds from the integer second
  */
  void setData(u8 * d, int dbytes, int dscan, int dsec, int dns);

 /**
  * reset all pcal objects
  */
  void resetpcal();
  void finalisepcal();

  /**
  * Calculates fringe rotation and fractional sample correction arrays and FFTs, and autocorrelates
  * @param index The index of the FFT chunk to process
  * @param subloopindex The "subloop" index to put the output in
  */
  virtual void process(int index, int subloopindex);

  virtual int process_gpu(int fftloop, int numBufferedFFTs, int startblock, int numblocks) { return 0; };

 /**
  * Sets the autocorrelation arrays to contain 0's
  */
  void zeroAutocorrelations();

 /**
  * Sets the kurtosis arrays to contain 0's
  */
  void zeroKurtosis();

 /**
  * Stores the times for the first FFT chunk to be processed
  * @param scan The current scan
  * @param seconds The offset in seconds from the start of the scan
  * @param ns The offset in nanoseconds from the integer second
  */
  void setOffsets(int scan, int seconds, int ns);

 /**
  * Averages the autocorrelations down in frequency
  */
  void averageFrequency();

 /**
  * Calculates the kurtosis from intermediate products and averages down in frequency if requires
  * @param numblocks The number of FFTs that went into this average
  * @param maxchannels The number of channels to average down to
  * @return True if there are valid kurtosis data
  */
  bool calculateAndAverageKurtosis(int numblocks, int maxchannels);

 /**
  * Grabs the pointer to an autocorrelation array
  * @param crosspol Whether to return the crosspolarisation autocorrelation for this band
  * @param outputband The band index
  */
  inline cf32* getAutocorrelation(bool crosspol, int outputband) const { return autocorrelations[(crosspol)?1:0][outputband]; }

 /**
  * Grabs the pointer to a kurtosis array
  * @param outputband The band index
  */
  inline f32* getKurtosis(int outputband) const { return sk[outputband]; }

 /**
  * Grabs the weight for a given band
  * @param crosspol Whether to return the crosspolarisation autocorrelation for this band
  * @param outputband The band index
  */
  inline f32 getWeight(bool crosspol, int outputband) const { return weights[(crosspol)?1:0][outputband]; }

 /**
  * Grabs the data weight for a given band
  * @param outputband The band index
  * @param subloopindex The index into the number of buffered FFTs that were processed in one batch
  */
  inline f32 getDataWeight(int outputband, int subloopindex) const { return perbandweights ? perbandweights[subloopindex][outputband] : dataweight[subloopindex]; }

 /**
  * Gets the expected decorrelation ("van Vleck correction" ) for a given number of bits.
  * All cases other than 1 and 2 are approximate only!!!
  * @param nbits The number of bits
  * @return The square root of the decorrelation correction
  */
  static inline float getDecorrelationPercentage(int nbits) { return decorrelationpercentage[nbits-1]; }

 /**
  * @return Whether this Mode is writing cross-polarisation auto-corelations
  */
  inline bool writeCrossAutoCorrs() const { return calccrosspolautocorrs; }

 /**
  * @return Whether this Mode was initialied ok
  */
  inline bool initialisedOK() const { return initok; }

 /**
  * @param dk Whether to calculate the kurtosis or not
  */
  inline void setDumpKurtosis(bool dk) { dumpkurtosis = dk; }

 /**
  * Returns a pointer to the FFT'd data of the specified product
  * @param outputband The band to get
  * @param subloopindex The "subloop" index to get the visibilities from
  * @return Pointer to the FFT'd data (complex 32 bit float)
  */
  inline const cf32* getFreqs(int outputband, int subloopindex) const { return fftoutputs[outputband][subloopindex]; };
  virtual const cuFloatComplex* getGpuFreqs() const { return nullptr; };

 /**
  * Returns a pointer to the FFT'd and conjugated data of the specified product
  * @param outputband The band to get
  * @param subloopindex The "subloop" index to get the visibilities from
  * @return Pointer to the conjugate of the FFT'd data (complex 32 bit float)
  */
  inline const cf32* getConjugatedFreqs(int outputband, int subloopindex) const { return conjfftoutputs[outputband][subloopindex]; }
  virtual const cuFloatComplex* getGpuConjugatedFreqs() const { return nullptr; };

 /**
  * Returns the estimated number of bytes used by the Mode
  * @return Estimated memory size of the Mode (bytes)
  */
  inline long long getEstimatedBytes() const { return estimatedbytes; }

  virtual ~Mode();

  /** Constant for comparing two floats for equality (for freqs and bandwidths etc) */
  static const float TINY;

  /**
   * Returns a single pcal result.
   * @param outputband The band to get
   * @param tone The number of the tone to get
   * @return pcal result
   */
  inline cf32 getPcal(int outputband, int tone) const
  { 
    return pcalresults[outputband][tone]; 
  }
  
  ///constant indicating no valid data in a subint
  static const int INVALID_SUBINT = -99999999;

protected:
 /**
  * Constructor: allocates memory, extracts stream information and calculates number of lookups etc
  * Note this is protected because you must instantiate either a CPUMode or a
  * GPUMode class -- you can't instantiate a bare Mode
  * @param conf The configuration object, containing all information about the duration and setup of this correlation
  * @param confindex The index of the configuration this Mode is for
  * @param dsindex The index of the datastream this Mode is for
  * @param recordedbandchan The number of channels for each recorded subband
  * @param chanstoavg The number of channels to average for each subband
  * @param bpersend The number of FFT blocks to be processed in a message
  * @param gsamples The number of additional guard samples at the end of a message
  * @param nrecordedfreqs The number of recorded frequencies for this Mode
  * @param recordedbw The bandwidth of each of these IFs
  * @param recordedfreqclkoffs The time offsets in microseconds to be applied post-F for each of the frequencies
  * @param recordedfreqclkoffsdelta The delay offsets in microseconds between Rcp and Lcp
  * @param recordedfreqphaseoffs The phase offsets in degrees between Rcp and Lcp
  * @param recordedfreqlooffs The LO offsets in Hz for each recorded frequency
  * @param nrecordedbands The total number of subbands recorded
  * @param nzoombands The number of subbands to be taken from within the recorded bands - can be zero
  * @param nbits The number of bits per sample
  * @param sampling The bit sampling type (real/complex)
  * @param tcomplex Type of complex sampling (single or double sideband)
  * @param unpacksamp The number of samples to unpack in one hit
  * @param fbank Whether to use a polyphase filterbank to channelise (instead of FFT)
  * @param linear2circular Whether to do a linear to circular conversion after the FFT
  * @param fringerotorder The interpolation order across an FFT (Oth, 1st or 2nd order; 0th = post-F)
  * @param arraystridelen The number of samples to stride when doing complex multiplies to implement sin/cos operations efficiently
  * @param cacorrs Whether cross-polarisation autocorrelations are to be calculated
  * @param bclock The recorder clock-out frequency in MHz ("block clock")
  */

  Mode(Configuration * conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend, int gsamples, int nrecordedfreqs, double recordedbw, double * recordedfreqclkoffs, double * recordedfreqclkoffsdelta, double * recordedfreqphaseoffs, double * recordedfreqlooffs, int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling, Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular, int fringerotorder, int arraystridelen, bool cacorrs, double bclock);

 /** 
  * Unpacks quantised data to float arrays.  The floating point values filled should
  * be in the range 0.0->1.0, and set appropriately to the expected input levels such that
  * the mean autocorrelation level at nominal sampler statistics is 0.??
  * @param sampleoffset The offset in number of time samples into the data array
  * @param subloopindex The "subloop" index that is currently being unpacked for (need to know to save weights in the right place)
  * @return The number of good samples unpacked scaled by the number of samples asked to unpack
  *         ie a weight in the range 0.0 to 1.0
  */
  virtual float unpack(int sampleoffset, int subloopindex);
  
  Configuration * config;
  int configindex, datastreamindex, recordedbandchannels, channelstoaverage, blockspersend, guardsamples, fftchannels, numrecordedfreqs, numrecordedbands, numzoombands, numbits, bytesperblocknumerator, bytesperblockdenominator, currentscan, offsetseconds, offsetns, order, flag, fftbuffersize, unpacksamples, unpackstartsamples, datasamples, avgdelsamples;
  long long estimatedbytes;
  int fringerotationorder, arraystridelength, numfrstrides, numfracstrides;
  double recordedbandwidth, blockclock, sampletime; //MHz, microseconds
  double a0, b0, c0, a, b, c, quadadd1, quadadd2;
  double fftstartmicrosec, fftdurationmicrosec, intclockseconds;
  f32 * dataweight;
  f32 ** perbandweights;
  int samplesperblock, samplesperlookup, numlookups, flaglength, autocorrwidth;
  int datascan, datasec, datans, datalengthbytes, usecomplex, usedouble;
  bool filterbank, calccrosspolautocorrs, fractionalLoFreq, initok, isfft, linear2circular;
  double * recordedfreqclockoffsets;
  double * recordedfreqclockoffsetsdelta;
  double * recordedfreqphaseoffset;
  double * recordedfreqlooffsets;
  bool deltapoloffsets, phasepoloffset;
  u8  *   data;
  s16 *   lookup;
  s16 *   linearunpacked;
  cf32*** fftoutputs;
  cf32*** conjfftoutputs;
  f32 **  weights;
  s32 *   validflags;
  cf32*** autocorrelations;
  vecFFTSpecR_f32 * pFFTSpecR;
  vecDFTSpecR_f32 * pDFTSpecR;
  u8 * fftbuffer;
  vecHintAlg hint;
  Model * model;
  f64 * interpolator;

  //new arrays for strided complex multiply for fringe rotation and fractional sample correction
  cf32 * complexrotator;
  cf32 * complexunpacked;
  cf32 * fracsamprotatorA, * fracsamprotatorB;  // Allow different delay correction for each pol
  cf32 * fftd;

  // variables for pcal
  int * pcalnbins;
  cf32 ** pcalresults;
  PCal ** extractor;
  
  f64 * subtoff;
  f64 * subtval;
  f64 * subxoff;
  f64 * subxval;
  f64 * subphase;
  f32 * subarg;
  f32 * subsin;
  f32 * subcos;

  f64 * steptoff;
  f64 * steptval;
  f64 * stepxoff;
  f64 * stepxval;
  f64 * stepphase;
  f32 * steparg;
  f32 * stepsin;
  f32 * stepcos;
  cf32 * stepcplx;

  f32 * subchannelfreqs;
  f32 * ldsbsubchannelfreqs;
  f32 * subfracsamparg;
  f32 * subfracsampsin;
  f32 * subfracsampcos;

  f32 * stepchannelfreqs;
  f32 * lsbstepchannelfreqs;
  f32 * dsbstepchannelfreqs;
  f32 * ldsbstepchannelfreqs;
  f32 * stepfracsamparg;
  f32 * stepfracsampsin;
  f32 * stepfracsampcos;
  cf32 * stepfracsampcplx;

  //extras necessary for quadratic (order == 2)
  cf32 * piecewiserotator;
  cf32 * quadpiecerotator;

  f64 * subquadxval;
  f64 * subquadphase;
  f32 * subquadarg;
  f32 * subquadsin;
  f32 * subquadcos;

  f64 * stepxoffsquared;
  f64 * tempstepxval;

  //kurtosis-specific variables
  bool dumpkurtosis;
  f32 *  kscratch; //[recordedbandchannels]
  f32 ** s1; //[numrecordedbands][recordedbandchannels]
  f32 ** s2; //[numrecordedbands][recordedbandchannels]
  f32 ** sk; //[numrecordedbands][recordedbandchannels]

  // Linear to circular conversion

  cf32 *phasecorrA, *phasecorrconjA, *phasecorrB, *phasecorrconjB; // 90 degrees + phase correction
  cf32 * tmpvec; 

private:
  ///Array containing decorrelation percentages for a given number of bits
  static const float decorrelationpercentage[];
};


#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
