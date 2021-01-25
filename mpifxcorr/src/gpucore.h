#ifndef GPUCORE_H
#define GPUCORE_H

#include "core.h"

class GPUCore: public Core {
	public:
		GPUCore(const int id, Configuration *const conf, int *const dids, MPI_Comm rcomm)
			: Core(id, conf, dids, rcomm) {};

		virtual void execute();
};

#endif
