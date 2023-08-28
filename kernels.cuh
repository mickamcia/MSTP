#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <thrust/extrema.h>

#define DIM 2048

#define SCALE_COUNT 9

__global__ void init_mat(float *d_signal, float rad, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    if((DIM / 2 - x - 0.5f) * (DIM / 2 - x - 0.5f) + (DIM / 2 - y - 0.5f) * (DIM / 2 - y - 0.5f) < rad * rad){
        d_signal[((x + DIM / 2) % DIM) * DIM + (y + DIM / 2) % DIM] = 1;
    }
    else{
        d_signal[((x + DIM / 2) % DIM) * DIM + (y + DIM / 2) % DIM] = 0;
    }
}
__global__ void multiply_mat(cufftComplex *d_output, cufftComplex *d_signal, cufftComplex *d_kernel, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    d_output[index] = cuCmulf(d_signal[index], d_kernel[index]);
}
__global__ void multiply_mat(cufftComplex *d_signal, cufftComplex *d_kernel, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    d_signal[index] = cuCmulf(d_signal[index], d_kernel[index]);
}
__global__ void divide_val(float *d_signal, float div, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    d_signal[index] = d_signal[index] / div;
}

__global__ void normalize_mat(float *d_signal, float min, float max, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    d_signal[index] = (d_signal[index] - min) / (max - min);
}
__global__ void subtract_abs_mat(float *d_output, float *d_in1, float *d_in2, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    d_output[index] = abs(d_in1[index] - d_in2[index]);
}
__global__ void calculate_smallest_scale(int *d_colors, float *d_variations, int* calculation_vector, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    int smallest_index = -1;
    for(int s = 0; s < SCALE_COUNT; s++){
        if(calculation_vector[s]){
            if(smallest_index == -1){
                smallest_index = s;
            }
            else{
                smallest_index = d_variations[smallest_index * width * height + index] < d_variations[s * width * height + index] ? smallest_index : s;
            }
        }
    }
    d_colors[index] = smallest_index;
}
__global__ void calculate_MSTP(float *d_output, int* d_colors, float *d_activators, float *d_inhibitors, float* modifiers_values, int* calculation_vector, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    int scale = d_colors[index];
    if(scale == -1) return;
    if(d_activators[scale * width * height + index] > d_inhibitors[scale * width * height + index]){
        d_output[index] = d_output[index] + modifiers_values[scale];
    }
    else{
        d_output[index] = d_output[index] - modifiers_values[scale];
    }
}
__global__ void calculate_texture(unsigned char *d_output, unsigned char* d_map, int* d_colors, float *d_values, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int index = y * width + x;
    if(d_colors[index] == -1) return;
    d_output[index * 4 + 0] = d_map[d_colors[index] * 4 + 0] * 0.99f * d_values[index] + (d_values[index] * 0.01f * 255);
    d_output[index * 4 + 1] = d_map[d_colors[index] * 4 + 1] * 0.99f * d_values[index] + (d_values[index] * 0.01f * 255);
    d_output[index * 4 + 2] = d_map[d_colors[index] * 4 + 2] * 0.99f * d_values[index] + (d_values[index] * 0.01f * 255);
    d_output[index * 4 + 3] = 255;
}