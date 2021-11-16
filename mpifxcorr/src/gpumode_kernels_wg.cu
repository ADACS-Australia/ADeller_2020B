#include "gpumode_kernels_wg.cuh"

void inner_loop()
{
    for (int j = 0; j < numrecordedbands; j++) // Loop over all recorded bands looking for the matching frequency we should be dealing with
    {
        if (config->matchingRecordedBand(configindex, datastreamindex, i, j))
        {
            indices[count++] = j;


            // Only deal with linear fringe rotation order
            if (usecomplex)
            {
                status = vectorMul_cf32(complexrotator, &unpackedcomplexarrays[j][nearestsample - unpackstartsamples], complexunpacked, fftchannels);
                // The following can be uncommented (and the above commented) if wanting to 'turn off' fringe rotation for testing in the complex case
                //status = vectorCopy_cf32(&unpackedcomplexarrays[j][nearestsample - unpackstartsamples], complexunpacked, fftchannels);
                if (status != vecNoErr)
                    csevere << startl << "Error in complex fringe rotation" << endl;
            }
            else
            {
                status = vectorRealToComplex_f32(&(unpackedarrays[j][nearestsample - unpackstartsamples]), NULL, complexunpacked, fftchannels);
                if (status != vecNoErr)
                    csevere << startl << "Error in real->complex conversion" << endl;
                status = vectorMul_cf32_I(complexrotator, complexunpacked, fftchannels);
                if (status != vecNoErr)
                    csevere << startl << "Error in fringe rotation!!!" << status << endl;
            }

            // TODO - Perform cuFFT (complexunpacked, fftd, plan, fftbuffer)
            gpuFFT_CtoC_cf32()

            // if (isfft)
            // {
            //     status = vectorFFT_CtoC_cf32(complexunpacked, fftd, pFFTSpecC, fftbuffer);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error doing the FFT!!!" << endl;
            // }
            // else
            // {
            //     status = vectorDFT_CtoC_cf32(complexunpacked, fftd, pDFTSpecC, fftbuffer);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error doing the DFT!!!" << endl;
            // }


            // NOT SUPPORTED - lower side band

            if (config->getDRecordedLowerSideband(configindex, datastreamindex, i))
            {
                // // All lower sideband bands need to be conjugated (achieved by taking the second half of the band for real-valued inputs)
                // // Additionally for the complex-valued inputs, the order of the frequency channels is reversed so they need to be flipped
                // // (for the double sideband case, in two halves, for the regular case, the whole thing)
                // if (usecomplex)
                // {
                //     if (usedouble)
                //     {
                //         status = vectorConjFlip_cf32(fftd, fftoutputs[j][subloopindex], recordedbandchannels / 2 + 1);
                //         status = vectorConjFlip_cf32(&fftd[recordedbandchannels / 2] + 1, &fftoutputs[j][subloopindex][recordedbandchannels / 2] + 1, recordedbandchannels / 2 - 1);
                //     }
                //     else
                //     {
                //         //status = vectorConjFlip_cf32(fftd, fftoutputs[j][subloopindex], recordedbandchannels);
                //         // note: using vectorConjFlip_cf32() -lofreq breaks Complex LSB (non-DSB!) fringes for VGOS *assuming* VGOS RDBE-G indeed LSB like memos claim
                //         // fix?: LSB fringes are restored at least for a synthetic fully correlated data set of Complex USB and Complex LSB data.
                //         //       The reversal has to be changed as below to retain DC in bin 0, producing not [ch1 ch2 ch3 ... DC] but instead [DC ch1 ch2 ch3 ...]
                //         // todo: validate fix on real world definitely-known-LSB data (evidenced by pcal tone positions etc), then uncomment the next lines:
                //         status = vectorConjFlip_cf32(fftd + 1, fftoutputs[j][subloopindex] + 1, recordedbandchannels - 1);
                //         fftoutputs[j][subloopindex][0] = fftd[0];
                //     }
                // }
                // else
                // {
                //     status = vectorCopy_cf32(&(fftd[recordedbandchannels]), fftoutputs[j][subloopindex], recordedbandchannels);
                // }
            }
            else
            {
                // For upper sideband bands, normally just need to copy the fftd channels.
                // However for complex double upper sideband, the two halves of the frequency space are swapped, so they need to be swapped back
                if (usecomplex && usedouble)
                {
                    // status = vectorCopy_cf32(fftd, &fftoutputs[j][subloopindex][recordedbandchannels / 2], recordedbandchannels / 2);
                    // status = vectorCopy_cf32(&fftd[recordedbandchannels / 2], fftoutputs[j][subloopindex], recordedbandchannels / 2);
                }
                else
                {
                    status = vectorCopy_cf32(fftd, fftoutputs[j][subloopindex], recordedbandchannels);
                }
            }
            if (status != vecNoErr)
                csevere << startl << "Error copying FFT results!!!" << endl;
            break;

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

            // if (dumpkurtosis) //do the necessary accumulation
            // {
            //     status = vectorMagnitude_cf32(fftoutputs[j][subloopindex], kscratch, recordedbandchannels);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error taking kurtosis magnitude!" << endl;
            //     status = vectorSquare_f32_I(kscratch, recordedbandchannels);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error in first kurtosis square!" << endl;
            //     status = vectorAdd_f32_I(kscratch, s1[j], recordedbandchannels);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error in kurtosis s1 accumulation!" << endl;
            //     status = vectorSquare_f32_I(kscratch, recordedbandchannels);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error in second kurtosis square!" << endl;
            //     status = vectorAdd_f32_I(kscratch, s2[j], recordedbandchannels);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error in kurtosis s2 accumulation!" << endl;
            // }

            // //do the frac sample correct (+ phase shifting if applicable, + fringe rotate if its post-f)
            // if (deltapoloffsets == false || config->getDRecordedBandPol(configindex, datastreamindex, j) == 'R')
            // {
            //     status = vectorMul_cf32_I(fracsamprotatorA, fftoutputs[j][subloopindex], recordedbandchannels);
            // }
            // else
            // {
            //     status = vectorMul_cf32_I(fracsamprotatorB, fftoutputs[j][subloopindex], recordedbandchannels);
            // }
            // if (status != vecNoErr)
            //     csevere << startl << "Error in application of frac sample correction!!!" << status << endl;

            // //do the conjugation
            // status = vectorConj_cf32(fftoutputs[j][subloopindex], conjfftoutputs[j][subloopindex], recordedbandchannels);
            // if (status != vecNoErr)
            //     csevere << startl << "Error in conjugate!!!" << status << endl;

            // if (!linear2circular)
            // {
            //     //do the autocorrelation (skipping Nyquist channel)
            //     status = vectorAddProduct_cf32(fftoutputs[j][subloopindex], conjfftoutputs[j][subloopindex], autocorrelations[0][j], recordedbandchannels);
            //     if (status != vecNoErr)
            //         csevere << startl << "Error in autocorrelation!!!" << status << endl;

            //     //store the weight for the autocorrelations
            //     if (perbandweights)
            //     {
            //         weights[0][j] += perbandweights[subloopindex][j];
            //     }
            //     else
            //     {
            //         weights[0][j] += dataweight[subloopindex];
            //     }
            // }
        }
    }
}