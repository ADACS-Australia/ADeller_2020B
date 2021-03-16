#include "gpumode_kernels.cuh"

#include <cuComplex.h>
#include <math.h>

#define TWO_PI                   6.283185307179586476925286766559


__global__ void _cudaMul_f64(const double *const src, const double by, double *const dest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if(idx > len) return;
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
  checkCuda(cudaMemset(dst, 0x0, len*sizeof(cuFloatComplex)));
  checkCuda(cudaMemcpy2D(dst, sizeof(cuFloatComplex), src, sizeof(float), sizeof(float), len, cudaMemcpyHostToDevice));
}

__global__ void _gpu_complexrotatorMultiply(cuFloatComplex *const a, const
    double bigA, const double bigB, cuFloatComplex *comparison) {
  const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  double exponent = ( bigA*j + bigB );
  exponent -= int(exponent);
  cuFloatComplex cr;
  sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
  const double max_re = (fabs(cr.x) > fabs(comparison[j].x) ? cr.x : comparison[j].x);
  const double max_im = (fabs(cr.y) > fabs(comparison[j].y) ? cr.y : comparison[j].y);
  const double diff_re = (cr.x - comparison[j].x)/max_re;
  const double diff_im = (cr.y - comparison[j].y)/max_im;
  /*
  if(fabs(diff_re) > 1e-2 || fabs(diff_im) > 1e-2) {
    printf("j = % 4d; cr = %0.17f + %0.17f ; c.f. orig = %0.17f + %0.17f\n"
           "   diff: %0.17e + %0.17e\n", j, cr.x, cr.y, comparison[j].x,
           comparison[j].y, diff_re, diff_im);
  }
  */
  //if(1 || diff_re > 0.0 || diff_im > 0.0) {
  //if(fabs(diff_re) > 1e-4 || fabs(diff_im) > 1e-4) {
  //if(cr.x != comparison[j].x || cr.y != comparison[j].y) {
    //a[j] = cuCmulf(a[j], comparison[j]);
  //} else {
    a[j] = cuCmulf(a[j], cr);
  //}
}

void gpu_complexrotatorMultiply(const size_t len, cuFloatComplex *const a,
    const double bigA, const double bigB, cuFloatComplex *comparison) {
  _gpu_complexrotatorMultiply<<<1,len>>>(a, bigA, bigB, comparison);
}

void *gpu_malloc(const size_t amt) {
  void *rv;
  checkCuda(cudaMalloc(&rv, amt));
  return rv;
}

// vim: shiftwidth=2:softtabstop=2:expandtab
