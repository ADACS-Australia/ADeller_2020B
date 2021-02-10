#include "gpumode_kernels.cuh"

__global__ void _cudaMul_f64(const double *const src, const double by, double *const dest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx > 16) return;
  dest[idx] = src[idx] * by;
}

void cudaMul_f64(const size_t len, const double *const src, const double by, double *const dest) {
  _cudaMul_f64<<<1,len>>>(src, by, dest);
}

// vim: shiftwidth=2:softtabstop=2:expandtab
