#include "mode_gpu.cuh"
#include <iostream>

using namespace std;

__global__ void cudaMul_cf32(cf32 *src1, cf32 *src2, cf32 *dest, int length) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += stride) {
        dest[i].x = src1[i].x * src2[i].x - src1[i].y * src2[i].y;
        dest[i].y = src1[i].x * src2[i].y + src1[i].y * src2[i].x;
    }
}

__global__ void cudaMul_cf32_I(cf32 *src, cf32 *srcdest, int length) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += stride) {
        auto tmp = srcdest[i].x * src[i].x - srcdest[i].y * src[i].y;
        srcdest[i].y = srcdest[i].x * src[i].y + srcdest[i].y * src[i].x;
        srcdest[i].x = tmp;
    }
}

__global__ void cudaConj_cf32(cf32 *src, cf32 *dest, int length) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += stride) {
        dest[i].x = src[i].x;
        dest[i].y = -src[i].y;
    }
}

__global__ void cudaAddProduct_cf32(cf32 *src1, cf32 *src2, cf32 *accumulator, length) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += stride) {
        accumulator[i].x += src1[i].x * src2[i].x - src1[i].y * src2[i].y;
        accumulator[i].y += src1[i].x * src2[i].y + src1[i].y * src2[i].x;
    }
}

int process(cf32 *complexrotator, cf32 *unpackedcomplex, cf32 *fftd, cf32 *fracsamprotator, int length, cufftHandle plan) {
    cf32 *d_complexrotator;
    cf32 *d_unpackedcomplex;
    cf32 *d_fftd;
    cf32 *d_fracsamprotator;
    cf32 *d_fftdconj;

    cudaMalloc((void **)&d_complexrotator, length * sizeof(cf32));
    cudaMalloc((void **)&d_unpackedcomplex, length * sizeof(cf32));
    cudaMalloc((void **)&d_fftd, length * sizeof(cf32));
    cudaMalloc((void **)&d_fftdconj, length * sizeof(cf32));
    cudaMalloc((void **)&d_fracsamprotator, length * sizeof(cf32));

    // 2 - copy input to device
    cudaMemcpy(d_unpackedcomplex, unpackedcomplex, length*sizeof(cf32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_complexrotator, complexrotator, length*sizeof(cf32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fracsamprotator, fracsamprotator, length*sizeof(cf32), cudaMemcpyHostToDevice);

    // 3 - perform the multiplication
    dim3 dimGrid(length / BLOCK_SIZE + 1);
    dim3 dimBlock(BLOCK_SIZE);
    cudaMul_cf32<<<dimGrid, dimBlock>>>(d_complexrotator, d_unpackedcomplex, d_fftd, length);

    // 4 - perform the FFT
    cufftExecC2C(plan, d_fftd, d_fftd, CUFFT_FORWARD);

    // 5 - perform fracsample rotate
    cudaMul_cf32_I<<<dimGrid, dimBlock>>>(d_fracsamprotator, d_fftd, length);

    // 6 - calc conjugate
    cudaConj_cf32<<<dimGrid, dimBlock>>>(d_fftd, d_fftdconj, length);

    // X - copy device to host
    cudaMemcpy(fftd, d_fftd, length * sizeof(cf32), cudaMemcpyDeviceToHost);

    // Y - synchronize - if using streams we should sync on that
    cudaDeviceSynchronize();

    // Z - free device memory
    cudaFree(d_fracsamprotator);
    cudaFree(d_complexrotator);
    cudaFree(d_unpackedcomplex);
    cudaFree(d_fftd);
    cudaFree(d_fftdconj);

    return 0;
}

#define TWO_PI 6.2831853072f
#define DEG_TO_RAD 0.03490658504f
#define FFT_LENGTH 512

int main() {
    cufftHandle plan;
    cufftPlan1d(&plan, FFT_LENGTH, CUFFT_C2C, 1);

    cf32 *complexrotator;
    complexrotator = (cf32 *)malloc(FFT_LENGTH * sizeof(cf32));

    cf32 *unpackedcomplex;
    unpackedcomplex = (cf32 *)malloc(FFT_LENGTH * sizeof(cf32));

    cf32 *fftd;
    fftd = (cf32 *)malloc(FFT_LENGTH * sizeof(cf32));

    cf32 *fracsamprotator;
    fracsamprotator = (cf32 *)malloc(FFT_LENGTH * sizeof(cf32));

    // prepare data
    for (auto i = 0; i < FFT_LENGTH; i++) {
        complexrotator[i].x = cos(TWO_PI * i / 512);
        complexrotator[i].y = sin(2./3. * TWO_PI * i / 512 + 45 * DEG_TO_RAD);
        
        // it's just 1, the multiply should work
        unpackedcomplex[i].x = 1;
        unpackedcomplex[i].y = 0;

        // for fracsamprotator lets multiply by i
        fracsamprotator[i].x = 0;
        fracsamprotator[i].y = 1;
    }

    process(complexrotator, unpackedcomplex, fftd, fracsamprotator, FFT_LENGTH, plan);

    // get the data out
    for (auto i = 0; i < 512; i++) {
        cout << "fftd[" << i << "]: (" << fftd[i].x << ", " << fftd[i].y << ")" << endl;
    }

    free(fftd);
    free(unpackedcomplex);
    free(complexrotator);
    cufftDestroy(plan);

    return 0;
}
