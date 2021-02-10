#ifndef GPUMODE_KERNELS_H
#define GPUMODE_KERNELS_H

#include <iostream>

#define checkCuda(err) __checkCuda(err, (char *)__FILE__, __LINE__)
inline cudaError_t __checkCuda(cudaError_t err, char *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "Error in calling CUDA operation in " << file << " at line " << line << std::endl;
    std::cerr << "Error was " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  return err;
}

// CUDA kernels
void cudaMul_f64(const size_t len, const double *const src, const double by, double *const dest);




#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
