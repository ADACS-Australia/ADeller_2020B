#ifndef GPUCORE_H
#define GPUCORE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "core.h"

class GPUCore : public Core {
public:
  GPUCore(const int id, Configuration *const conf, int *const dids, MPI_Comm rcomm);

  virtual void loopprocess(int threadid);

  void processgpudata(int index, int threadid, int startblock, int numblocks, Mode **modes, Polyco *currentpolyco,
                      threadscratchspace *scratchspace);

protected:
  virtual Mode *getMode(const int configindex, const int datastreamindex) {
      return config->getMode(configindex, datastreamindex, true);
  }

private:
  void processBaselineBased(
          const cuFloatComplex* freqData,
          const cuFloatComplex* conjFreqData,
          char* stream1BandIndexes_gpu,
          char* stream2BandIndexes_gpu,
          cuFloatComplex* threadcrosscorrs_gpu,
          int xmacstridelength,
          int numPolarisationProducts,
          int numBufferedFFTs,
          int xmacstart,
          int resultindex,
          int fftloop,
          int startblock,
          int numblocks,
          int fftchannels,
          int numrecordedbands,
          cudaStream_t cuStream
  );

  int cudaMaxThreadsPerBlock;
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
