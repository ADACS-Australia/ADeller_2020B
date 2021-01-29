#ifndef GPUCORE_H
#define GPUCORE_H

#include "core.h"

class GPUCore: public Core {
  public:
    GPUCore(const int id, Configuration *const conf, int *const dids, MPI_Comm rcomm)
      : Core(id, conf, dids, rcomm) {};

    virtual void processdata(int index, int threadid, int startblock, int numblocks, Mode ** modes, Polyco * currentpolyco, threadscratchspace * scratchspace);

  protected:
    /*
    virtual Mode *getMode(const int configindex, const int datastreamindex) {
      return config->getMode(configindex, datastreamindex, true);
    }
    */
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
