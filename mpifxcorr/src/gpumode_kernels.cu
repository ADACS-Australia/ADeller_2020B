#include "gpumode_kernels.cuh"

__global__ void _cudaMul_f64(const double *const src, const double by, double *const dest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx > 16) return;
  dest[idx] = src[idx] * by;
}

void cudaMul_f64(const size_t len, const double *const src, const double by, double *const dest) {
  _cudaMul_f64<<<1,len>>>(src, by, dest);
}

__global__ void _gpu_inPlaceMultiply_cf(const cuFloatComplex *const src, cuFloatComplex *const dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = cuCmulf(dst[idx], src[idx]);
}

void gpu_inPlaceMultiply_cf(const cuFloatComplex *const dst, cuFloatComplex *const bydst, const size_t len) {
  _gpu_inPlaceMultiply_cf<<<1,len>>>(dst, bydst);
}

void gpu_host2DevRtoC(cuFloatComplex *const dst, const float *const src, const size_t len) {
  checkCuda(cudaMemset(dst, 0x0, len));
  checkCuda(cudaMemcpy2D(dst, sizeof(cuFloatComplex), src, sizeof(float), sizeof(float), len, cudaMemcpyHostToDevice));
}

void *gpu_malloc(const size_t amt) {
  void *rv;
  checkCuda(cudaMalloc(&rv, amt));
  return rv;
}

// vim: shiftwidth=2:softtabstop=2:expandtab
