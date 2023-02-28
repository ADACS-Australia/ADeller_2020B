#ifndef GPUMODE_KERNELS_H
#define GPUMODE_KERNELS_H

#include <iostream>
#include <cuComplex.h>
#include <cufft.h>

#define checkCuda(err) __checkCuda(err, (char *)__FILE__, __LINE__)
inline cudaError_t __checkCuda(cudaError_t err, char *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "Error in calling CUDA operation in " << file << " at line " << line << std::endl;
    std::cerr << "Error was " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  return err;
}

#define checkCufft(err) __checkCufft(err, (char *)__FILE__, __LINE__)
inline cufftResult_t __checkCufft(const cufftResult_t err, const char *const file, const int line) {
  if (err != CUFFT_SUCCESS) {
    std::cerr << "Error calling a cuFFT operation in " << file << " at line " << line << std::endl;
    // TODO: should we convert err to a string? (it is an enum documented in
    // the cuFFT documentation - there doesn't seem to be an official errorcode
    // -> string conversion routine, but there is _cudaGetErrorEnum...)
    std::cerr << "Error was " << (int)err << std::endl;
    exit(1);
  }
  return err;
}

// CUDA kernels
void cudaMul_f64(const size_t len, const double *const src, const double by, double *const dest);

// gpu_inPlaceMultiply_cf(complexrotator_gpu, complexunpacked_gpu, fftchannels);
/**
 * In-place complex multiplication. Multiply src by 'bydest' and store in
 * 'bydest'.
 * @param src Source buffer
 * @param dst Destination buffer
 * @param len Number of samples to multiply. */
void gpu_inPlaceMultiply_cf(const cuFloatComplex *const src, cuFloatComplex *const bydst, const size_t len);

/**
 * Copy a real, host buffer to a complex, device buffer, initalising the
 * imaginary components to zero.
 * @param src Source buffer
 * @param dst Destination buffer
 * @param len Number of samples to copy
 */
void gpu_host2DevRtoC(cuFloatComplex *const dst, const float *const src, const size_t len);


#define NOT_SUPPORTED(x) { std::cerr << "Whoops, we don't support this on the GPU: " << x << std::endl; exit(1); }

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
