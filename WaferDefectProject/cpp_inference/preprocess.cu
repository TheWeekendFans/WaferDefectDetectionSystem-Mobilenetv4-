#include <cuda_runtime.h>
#include <cstdint>

// Custom kernel for Normalization
// (x - mean) / std
// Mean: [0.485, 0.456, 0.406]
// Std:  [0.229, 0.224, 0.225]
__global__ void preprocess_kernel_nchw(const uint8_t* src, float* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // HWC to NCHW conversion and normalization
    // Input: RGBRGB...
    // Output: RRR...GGG...BBB...
    
    float r = src[idx * 3 + 0] / 255.0f;
    float g = src[idx * 3 + 1] / 255.0f;
    float b = src[idx * 3 + 2] / 255.0f;

    // Normalize
    dst[0 * height * width + idx] = (r - 0.485f) / 0.229f;
    dst[1 * height * width + idx] = (g - 0.456f) / 0.224f;
    dst[2 * height * width + idx] = (b - 0.406f) / 0.225f;
}

extern "C" void launch_preprocess_kernel(uint8_t* src, float* dst, int width, int height, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    
    preprocess_kernel_nchw<<<blocks, threads, 0, stream>>>(src, dst, width, height);
}
