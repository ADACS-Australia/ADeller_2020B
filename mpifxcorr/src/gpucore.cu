#include "gpucore.h"
#include "alert.h"
#include <iostream>

void GPUCore::processdata(int index, int threadid, int startblock, int numblocks, Mode ** modes, Polyco * currentpolyco, threadscratchspace * scratchspace)
{
  static int nth_call = 0;
  ++nth_call;
  std::cout << "called GPUCore::processdata for the " << nth_call << time << std::endl;
#ifndef NEUTERED_DIFX
  int status, i, numfftloops, numfftsprocessed;
  int resultindex, cindex, ds1index, ds2index, binloop;
  int xcblockcount, maxxcblocks, xcshiftcount;
  int acblockcount, maxacblocks, acshiftcount;
  int freqchannels;
  int xmacstridelength, xmacpasses, xmacstart, destbin, destchan, localfreqindex;
  int dsfreqindex;
  char papol;
  double offsetmins, blockns;
  f32 bweight;
  f64 * binweights;
  const Mode * m1, * m2;
  const cf32 * vis1;
  const cf32 * vis2;
  double sampletimens;
  int starttimens;
  int fftsize;
  int numBufferedFFTs;
  float weight1, weight2;
#endif
  int perr;

//following statement used to cut all all processing for "Neutered DiFX"
#ifndef NEUTERED_DIFX
  xmacstridelength = config->getXmacStrideLength(procslots[index].configindex);
  binloop = 1;
  if(procslots[index].pulsarbin && !procslots[index].scrunchoutput)
    binloop = procslots[index].numpulsarbins;
  if(procslots[index].pulsarbin)
    binweights = currentpolyco->getBinWeights();
  else
    binweights = 0;
  numBufferedFFTs = config->getNumBufferedFFTs(procslots[index].configindex);

  //set up the mode objects that will do the station-based processing
  for(int j=0;j<numdatastreams;j++)
  {
    //zero the autocorrelations and set delays
    modes[j]->zeroAutocorrelations();
    modes[j]->setValidFlags(&(procslots[index].controlbuffer[j][3]));
    modes[j]->setData(procslots[index].databuffer[j], procslots[index].datalengthbytes[j], procslots[index].controlbuffer[j][0], procslots[index].controlbuffer[j][1], procslots[index].controlbuffer[j][2]);
    modes[j]->setOffsets(procslots[index].offsets[0], procslots[index].offsets[1], procslots[index].offsets[2]);
    modes[j]->setDumpKurtosis(scratchspace->dumpkurtosis);
    if(scratchspace->dumpkurtosis)
      modes[j]->zeroKurtosis();
    
    //reset pcal
    if(config->getDPhaseCalIntervalMHz(procslots[index].configindex, j) > 0)
    {
      // Calculate the sample time. Every band has the same bandwidth.
      sampletimens = 1.0/(2.0*config->getDRecordedBandwidth(procslots[index].configindex, j, 0))*1e+3;
      
      // Get the ns start time of the whole block.
      starttimens = procslots[index].offsets[2];
      
      // Calculate the FFT size in number of samples.
      fftsize = 2*config->getFNumChannels(config->getDRecordedFreqIndex(procslots[index].configindex, j, 0));
      
      modes[j]->resetpcal();
    }
  }

  //zero the results for this thread
  status = vectorZero_cf32(scratchspace->threadcrosscorrs, procslots[index].threadresultlength);
  if(status != vecNoErr)
    csevere << startl << "Error trying to zero threadcrosscorrs!!!" << endl;

  //zero the baselineweights and baselineshiftdecorrs for this thread
  for(int i=0;i<config->getFreqTableLength();i++)
  {
    if(config->isFrequencyUsed(procslots[index].configindex, i))
    {
      for(int b=0;b<binloop;b++)
      {
        for(int j=0;j<numbaselines;j++)
        {
          localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, i);
          if(localfreqindex >= 0)
          {
            status = vectorZero_f32(scratchspace->baselineweight[i][b][j], config->getBNumPolProducts(procslots[index].configindex, j, localfreqindex));
            if(status != vecNoErr)
              csevere << startl << "Error trying to zero baselineweight[" << i << "][" << b << "][" << j << "]!!!" << endl;
          }
        }
      }
      if(model->getNumPhaseCentres(procslots[index].offsets[0]) > 1)
      {
        for(int j=0;j<numbaselines;j++)
        {
          localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, i);
          if(localfreqindex >= 0)
          {
            status = vectorZero_f32(scratchspace->baselineshiftdecorr[i][j], model->getNumPhaseCentres(procslots[index].offsets[0]));
            if(status != vecNoErr)
              csevere << startl << "Error trying to zero baselineshiftdecorr[" << i << "][" << j << "]!!!" << endl;
          }
        }
      }
    }
  }

  //set up variables which control the number of loops through buffered FFT results
  xcblockcount = 0;
  xcshiftcount = 0;
  acblockcount = 0;
  acshiftcount = 0;
  numfftloops = numblocks/numBufferedFFTs;
  if(numblocks%numBufferedFFTs != 0)
    numfftloops++;
  blockns = ((double)(config->getSubintNS(procslots[index].configindex)))/((double)(config->getBlocksPerSend(procslots[index].configindex)));

  maxxcblocks = ((int)(model->getMaxNSBetweenXCAvg(procslots[index].offsets[0])/blockns));
  maxxcblocks -= maxxcblocks%numBufferedFFTs;
  if(maxxcblocks == 0) {
    maxxcblocks = numBufferedFFTs;
    cverbose << startl << "Requested cross-correlation shift/average time of " << model->getMaxNSBetweenXCAvg(procslots[index].offsets[0]) << " ns cannot be met with " << numBufferedFFTs << " FFTs being buffered; the time resolution which will be attained is " << maxxcblocks*blockns << " ns" << endl;
  }

  maxacblocks = ((int)(model->getMaxNSBetweenACAvg(procslots[index].offsets[0])/blockns));
  maxacblocks -= maxacblocks%numBufferedFFTs;
  if(maxacblocks == 0) {
    maxacblocks = numBufferedFFTs;
    cverbose << startl << "Requested autocorrelation shift/average time of " << model->getMaxNSBetweenACAvg(procslots[index].offsets[0]) << " ns cannot be met with " << numBufferedFFTs << " FFTs being buffered; the time resolution which will be attained is " << maxacblocks*blockns << " ns" << endl;
  }

  //process each chunk of FFTs in turn
  for(int fftloop=0;fftloop<numfftloops;fftloop++)
  {
    numfftsprocessed = 0;   // not strictly needed, but to prevent compiler warning
    //do the station-based processing for this batch of FFT chunks
    for(int j=0;j<numdatastreams;j++)
    {
      numfftsprocessed = 0;
      for(int fftsubloop=0;fftsubloop<numBufferedFFTs;fftsubloop++)
      {
        i = fftloop*numBufferedFFTs + fftsubloop + startblock;
	if(i >= startblock+numblocks)
	  break; //may not have to fully complete last fftloop
        modes[j]->process_gpu(i, fftsubloop);
        numfftsprocessed++;
      }
    }

    //if necessary, work out the pulsar bins
    if(procslots[index].pulsarbin)
    {
      for(int fftsubloop=0;fftsubloop<numBufferedFFTs; fftsubloop++)
      {
        i = fftloop*numBufferedFFTs + fftsubloop + startblock;
        offsetmins = ((double)i)*blockns/60000000000.0;
        currentpolyco->getBins(offsetmins, scratchspace->bins[fftsubloop]);
      }
    }

    //do the baseline-based processing for this batch of FFT chunks
    resultindex = 0;
    for(int f=0;f<config->getFreqTableLength();f++)
    {
      if(config->phasedArrayOn(procslots[index].configindex)) //phased array processing
      {
        freqchannels = config->getFNumChannels(procslots[index].configindex);
        for(int j=0;j<config->getFPhasedArrayNumPols(procslots[index].configindex, f);j++)
        {
          papol = config->getFPhaseArrayPol(procslots[index].configindex, f, j);

          //weight and add the results for each baseline
          for(int fftsubloop=0;fftsubloop<numBufferedFFTs;fftsubloop++)
          {
            for(int k=0;k<numdatastreams;k++)
            {
              vis1 = 0;
              for(int l=0;l<config->getDNumRecordedBands(procslots[index].configindex, k);l++)
              {
                dsfreqindex = config->getDRecordedFreqIndex(procslots[index].configindex, k, l);
                if(dsfreqindex == f && config->getDRecordedBandPol(procslots[index].configindex, k, l) == papol) {
                  vis1 = modes[k]->getFreqs(l, fftsubloop);
                  break;
                }
              }
              if(vis1 == 0)
              {
                for(int l=0;l<config->getDNumZoomBands(procslots[index].configindex, k);l++)
                {
                  dsfreqindex = config->getDZoomFreqIndex(procslots[index].configindex, k, l);
                  if(dsfreqindex == f && config->getDZoomBandPol(procslots[index].configindex, k, l) == papol) {
                    vis1 = modes[k]->getFreqs(config->getDNumRecordedBands(procslots[index].configindex, k) + l, fftsubloop);
                    break;
                  }
                }
              }
              if(vis1 != 0)
              {
                //weight the data
                status = vectorMulC_f32((f32*)vis1, config->getFPhasedArrayDWeight(procslots[index].configindex, f, k), (f32*)scratchspace->rotated, freqchannels*2);
                if(status != vecNoErr)
                  cerror << startl << "Error trying to scale phased array results!" << endl;

                //add it to the result
                status = vectorAdd_cf32_I(scratchspace->rotated, &(scratchspace->threadcrosscorrs[resultindex]), freqchannels);
                if(status != vecNoErr)
                  cerror << startl << "Error trying to add phased array results!" << endl;
              }
            }
          }
          resultindex += freqchannels;
        }
      }
      else if(config->isFrequencyUsed(procslots[index].configindex, f)) //normal processing
      {
        //All baseline freq indices into the freq table are determined by the *first* datastream
        //in the event of correlating USB with LSB data.  Hence all Nyquist offsets/channels etc
        //are determined by the freq corresponding to the *first* datastream
        freqchannels = config->getFNumChannels(f);
        xmacpasses = config->getNumXmacStrides(procslots[index].configindex, f);
        for(int x=0;x<xmacpasses;x++)
        {
          xmacstart = x*xmacstridelength;

          //do the cross multiplication - gets messy for the pulsar binning
          for(int j=0;j<numbaselines;j++)
          {
            //get the localfreqindex for this frequency
            localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, f);
            if(localfreqindex >= 0)
            {
              //get the two modes that contribute to this baseline
              ds1index = config->getBOrderedDataStream1Index(procslots[index].configindex, j);
	      ds2index = config->getBOrderedDataStream2Index(procslots[index].configindex, j);
	      m1 = modes[ds1index];
	      m2 = modes[ds2index];

              //do the baseline-based processing for this batch of FFT chunks
              for(int fftsubloop=0;fftsubloop<numBufferedFFTs;fftsubloop++)
              {
	        i = fftloop*numBufferedFFTs + fftsubloop + startblock;
	        if(i >= startblock+numblocks)
		  break; //may not have to fully complete last fftloop

                //add the desired results into the resultsbuffer, for each polarisation pair [and pulsar bin]
                //loop through each polarisation for this frequency
                for(int p=0;p<config->getBNumPolProducts(procslots[index].configindex,j,localfreqindex);p++)
                {
                  //get the appropriate arrays to multiply
                  vis1 = &(m1->getFreqs(config->getBDataStream1BandIndex(procslots[index].configindex, j, localfreqindex, p), fftsubloop)[xmacstart]);
                  vis2 = &(m2->getConjugatedFreqs(config->getBDataStream2BandIndex(procslots[index].configindex, j, localfreqindex, p), fftsubloop)[xmacstart]);

                  weight1 = m1->getDataWeight(config->getBDataStream1RecordBandIndex(procslots[index].configindex, j, localfreqindex, p), fftsubloop);
                  weight2 = m2->getDataWeight(config->getBDataStream2RecordBandIndex(procslots[index].configindex, j, localfreqindex, p), fftsubloop);

                  if(procslots[index].pulsarbin)
                  {
                    //multiply into scratch space
                    status = vectorMul_cf32(vis1, vis2, scratchspace->pulsarscratchspace, xmacstridelength);
                    if(status != vecNoErr)
                      csevere << startl << "Error trying to xmac baseline " << j << " frequency " << localfreqindex << " polarisation product " << p << ", status " << status << endl;

                    //if scrunching, add into temp accumulate space, otherwise add into normal space
                    if(procslots[index].scrunchoutput)
                    {
                      bweight = weight1*weight2/freqchannels;
                      destchan = xmacstart;
                      for(int l=0;l<xmacstridelength;l++)
                      {
                        //the first zero (the source slot) is because we are limiting to one pulsar ephemeris for now
                        destbin = scratchspace->bins[fftsubloop][f][destchan];
                        scratchspace->pulsaraccumspace[f][x][j][0][p][destbin][l].re += scratchspace->pulsarscratchspace[l].re;
                        scratchspace->pulsaraccumspace[f][x][j][0][p][destbin][l].im += scratchspace->pulsarscratchspace[l].im;
                        scratchspace->baselineweight[f][0][j][p] += bweight*binweights[destbin];
                        destchan++;
                      }
                    }
                    else
                    {
                      bweight = weight1*weight2/freqchannels;
                      destchan = xmacstart;
                      for(int l=0;l<xmacstridelength;l++)
                      {
                        destbin = scratchspace->bins[fftsubloop][f][destchan];
                        //cindex = resultindex + (scratchspace->bins[freqindex][destchan]*config->getBNumPolProducts(procslots[index].configindex,j,localfreqindex) + p)*(freqchannels+1) + destchan;
                        cindex = resultindex + (destbin*config->getBNumPolProducts(procslots[index].configindex,j,localfreqindex) + p)*xmacstridelength + l;
                        scratchspace->threadcrosscorrs[cindex].re += scratchspace->pulsarscratchspace[l].re;
                        scratchspace->threadcrosscorrs[cindex].im += scratchspace->pulsarscratchspace[l].im;
                        scratchspace->baselineweight[f][destbin][j][p] += bweight;
                        destchan++;
                      }
                    }
                  }
                  else
                  {
                    //not pulsar binning, so this is nice and simple - just cross multiply accumulate
                    status = vectorAddProduct_cf32(vis1, vis2, &(scratchspace->threadcrosscorrs[resultindex+p*xmacstridelength]), xmacstridelength);

                    if(status != vecNoErr)
                      csevere << startl << "Error trying to xmac baseline " << j << " frequency " << localfreqindex << " polarisation product " << p << ", status " << status << endl;
                  }
                }
              }
	      if(procslots[index].pulsarbin && !procslots[index].scrunchoutput)
	        resultindex += config->getBNumPolProducts(procslots[index].configindex,j,localfreqindex)*procslots[index].numpulsarbins*xmacstridelength;
              else
	        resultindex += config->getBNumPolProducts(procslots[index].configindex,j,localfreqindex)*xmacstridelength;
            }
          }
        }
      }
    }

    xcblockcount += numfftsprocessed;
    if(xcblockcount == maxxcblocks)
    {
      //shift/average and then lock results and copy data
      uvshiftAndAverage(index, threadid, (startblock+xcshiftcount*maxxcblocks+((double)maxxcblocks)/2.0)*blockns, maxxcblocks*blockns, currentpolyco, scratchspace);
      //reset the xcblockcount, increment xcshiftcount
      xcblockcount = 0;
      xcshiftcount++;
    }
    acblockcount += numfftsprocessed;
    if(acblockcount == maxacblocks)
    {
      //shift/average and then lock results and copy data
      averageAndSendAutocorrs(index, threadid, (startblock+acshiftcount*maxacblocks+((double)maxacblocks)/2.0)*blockns, maxacblocks*blockns, modes, scratchspace);
      //reset the acblockcount, increment acshiftcount, zero the autocorrelations
      acblockcount = 0;
      acshiftcount++;
      for(int j=0;j<numdatastreams;j++)
        modes[j]->zeroAutocorrelations();
    }

    //finally, update the baselineweight if not doing any pulsar stuff
    if(!procslots[index].pulsarbin)
    {
      for(int fftsubloop=0;fftsubloop<numBufferedFFTs;fftsubloop++)
      {
        i = fftloop*numBufferedFFTs + fftsubloop + startblock;
        if(i >= startblock+numblocks)
          break; //may not have to fully complete last fftloop
        for(int f=0;f<config->getFreqTableLength();f++)
        {
          if(config->isFrequencyUsed(procslots[index].configindex, f))
          {
            for(int j=0;j<numbaselines;j++)
	    {
	      localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, f);
	      if(localfreqindex >= 0)
	      {
	        ds1index = config->getBOrderedDataStream1Index(procslots[index].configindex, j);
	        ds2index = config->getBOrderedDataStream2Index(procslots[index].configindex, j);
	        m1 = modes[ds1index];
	        m2 = modes[ds2index];
	        for(int p=0;p<config->getBNumPolProducts(procslots[index].configindex,j,localfreqindex);p++)
		{
                  int ds1recordbandindex, ds2recordbandindex;

                  ds1recordbandindex = config->getBDataStream1RecordBandIndex(procslots[index].configindex, j, localfreqindex, p);
                  ds2recordbandindex = config->getBDataStream2RecordBandIndex(procslots[index].configindex, j, localfreqindex, p);

                  if(ds1recordbandindex < 0 || ds2recordbandindex < 0)
                  {
                    cerror << startl << "Error: Core::processdata(): one of the record band indices could not be found: ds1recordbandindex = " << ds1recordbandindex << " ds2recordbandindex = " << ds2recordbandindex << endl;
                  }
                  else
                  {
                    weight1 = m1->getDataWeight(ds1recordbandindex, fftsubloop);
                    weight2 = m2->getDataWeight(ds2recordbandindex, fftsubloop);

                    bweight = weight1*weight2;

                    scratchspace->baselineweight[f][0][j][p] += bweight;
                  }
		}
	      }
	    }
	  }
        }
      }
    }
  }

  if(xcblockcount != 0) {
    uvshiftAndAverage(index, threadid, (startblock+xcshiftcount*maxxcblocks+((double)xcblockcount)/2.0)*blockns, xcblockcount*blockns, currentpolyco, scratchspace);
  }
  if(acblockcount != 0) {
    averageAndSendAutocorrs(index, threadid, (startblock+acshiftcount*maxacblocks+((double)acblockcount)/2.0)*blockns, acblockcount*blockns, modes, scratchspace);
  }
  if(scratchspace->dumpkurtosis) {
    averageAndSendKurtosis(index, threadid, (startblock + numblocks/2.0)*blockns, numblocks*blockns, numblocks, modes, scratchspace);
  }

  //lock the bweight copylock, so we're the only one adding to the result array (baseline weight section)
  perr = pthread_mutex_lock(&(procslots[index].bweightcopylock));
  if(perr != 0)
    csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying lock bweight copy mutex!!!" << endl;

  for(int f=0;f<config->getFreqTableLength();f++)
  {
    if(config->isFrequencyUsed(procslots[index].configindex, f))
    {
      for(int i=0;i<numbaselines;i++)
      {
        localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, i, f);
        if(localfreqindex >= 0)
        {
          resultindex = config->getCoreResultBWeightOffset(procslots[index].configindex, f, i)*2;
          for(int b=0;b<binloop;b++)
          {
            for(int j=0;j<config->getBNumPolProducts(procslots[index].configindex,i,localfreqindex);j++)
            {
              procslots[index].floatresults[resultindex] += scratchspace->baselineweight[f][b][i][j];
              resultindex++;
            }
          }
        }
      }
      if(model->getNumPhaseCentres(procslots[index].offsets[0]) > 1)
      {
        for(int i=0;i<numbaselines;i++)
        {
          localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, i, f);
          if(localfreqindex >= 0)
          {
            resultindex = config->getCoreResultBShiftDecorrOffset(procslots[index].configindex, f, i)*2;
            for(int s=0;s<model->getNumPhaseCentres(procslots[index].offsets[0]);s++)
            {
              procslots[index].floatresults[resultindex] += scratchspace->baselineshiftdecorr[f][i][s];
              resultindex++;
            }
          }
        }
      }
    }
  }

  //unlock the bweight copylock
  perr = pthread_mutex_unlock(&(procslots[index].bweightcopylock));
  if(perr != 0)
    csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying unlock copy mutex!!!" << endl;

  //copy the PCal results
  copyPCalTones(index, threadid, modes);

//end the cutout of processing in "Neutered DiFX"
#endif

  //grab the next slot lock
  perr = pthread_mutex_lock(&(procslots[(index+1)%RECEIVE_RING_LENGTH].slotlocks[threadid]));
  if(perr != 0)
    csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying lock mutex " << (index+1)%RECEIVE_RING_LENGTH << endl;

  //unlock the one we had
  perr = pthread_mutex_unlock(&(procslots[index].slotlocks[threadid]));
  if(perr != 0)
    csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying unlock mutex " << index << endl;
}
// vim: shiftwidth=2:softtabstop=2:expandtab