#include <iostream>
#include "kernels.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK {                                                                       \
 cudaError_t e=cudaGetLastError();                                                         \
 if(e!=cudaSuccess) {                                                                      \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0);                                                                                \
 }                                                                                         \
}
/* important init_mat 100
int activator_radii[SCALE_COUNT]    = {4,8,16,32,64,128,256,384,512,768};
int inhibitor_radii[SCALE_COUNT]    = {8,16,32,64,128,256,512,768,1024,1536};
int variation_radii[SCALE_COUNT]    = {1,1,1,1,1,1,1,1,1,1};
float modifiers_values[SCALE_COUNT] = {0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002};
int calculation_vector[SCALE_COUNT] = {1,1,1,1,1,1,1,1,1};
*/
int activator_radii[SCALE_COUNT]    = {4,16,64,256,1024};
int inhibitor_radii[SCALE_COUNT]    = {8,32,128,512,2048};
//int activator_radii[SCALE_COUNT]    = {64,128,256,512,1024};
//int inhibitor_radii[SCALE_COUNT]    = {128,256,512,1024,2048};
int variation_radii[SCALE_COUNT]    = {1,1,1,1,1};
float modifiers_values[SCALE_COUNT] = {0.002,0.002,0.002,0.002,0.002};
int calculation_vector[SCALE_COUNT] = {1,1,1,1,1};
/*
int activator_radii[SCALE_COUNT]    = {4,8,16,32,64,128,256};
int inhibitor_radii[SCALE_COUNT]    = {8,16,32,64,128,256,512};
int variation_radii[SCALE_COUNT]    = {1,1,1,1,1,1,1};
float modifiers_values[SCALE_COUNT] = {0.02,0.02,0.02,0.02,0.02,0.02,0.02};
int calculation_vector[SCALE_COUNT] = {0,0,0,0,0,0};
*/
/*
unsigned char color_map[SCALE_COUNT * 4] = {
    255,0,  0,  255,
    0,  255,0,  255,
    0,  0,  255,255,
    255,0,  0,  255,
    0,  255,0,  255,
    0,  0,  255,255,
    255,0,  0,  255,
    0,  255,0,  255,
    0,  0,  255,255,
    255,0,  0,  255,
    0,  255,0,  255,
    0,  0,  255,255,
    255,0,  0,  255,
    0,  255,0,  255,
    0,  0,  255,255,
    255,0,  0,  255,
};*/
/*
unsigned char color_map[SCALE_COUNT * 4] = {
    232,  215,  241,  255,  
    255,0,  0,  255,  
    48, 31,71,  255,  
    188,96,255,255,  
    255,255,0,  255,  
    255,0,  255,255,  
    0,  255,255,255,  
    13,19,33,255,  
    0,  0,  0,  255,
    161,103,165, 255,
    0,  255,0,  255,
    255,255,0,  255,
    74,48,108,255,
    255,0,  255,255,
    211,188,204,255,
    232,  215,  241,255,
};
*/
/*
unsigned char color_map[SCALE_COUNT * 4] = {
    255,77,77,0,
    255,165,0,0,
    255,255,0,0,
    127,255,0,0,
    0,255,0,0,
    0,255,127,0,
    0,255,255,0,
    0,127,255,0,
    0,0,255,0,
    127,0,255,0,
    255,0,255,0,
    255,0,127,0,
    139,69,19,0,
    128,128,128,0,
    0,191,255,0,
    255,20,147,0,
};
*/

/* //those look nice
unsigned char color_map[SCALE_COUNT * 4] = {
    248, 201, 119, 0,
    63, 136, 143, 0,
    198, 82, 69, 0,
    117, 183, 138, 0,
    235, 151, 78, 0,
    147, 118, 177, 0,
    87, 171, 191, 0,
    217, 127, 132, 0,
    168, 205, 136, 0,
};*/
//emma's

/*
unsigned char color_map[SCALE_COUNT * 4] = {
    227, 215, 255, 0,
    175, 162, 255, 0,
    236, 220, 11, 0,
    244, 127, 59, 0,
    209, 48, 201, 0,
    227, 215, 255, 0,
    175, 162, 255, 0,
    236, 220, 11, 0,
    244, 127, 59, 0,
    209, 48, 201, 0,
};
*/
unsigned char color_map[SCALE_COUNT * 4] = {
    //idk 0, 128, 128, 0, 255, 107, 107, 0, 143, 182, 140, 0, 107, 62, 131, 0, 255, 209, 102, 0
    0, 102, 204, 0, 0, 128, 0, 0, 153, 102, 204, 0, 64, 64, 64, 0, 150, 113, 23, 0
    //this 227, 215, 255, 0, 175, 162, 255, 0, 236, 220, 11, 0, 244, 127, 59, 0, 209, 48, 201, 0,
};
/*
unsigned char color_map[SCALE_COUNT * 4] = {
    220, 70, 89, 0,
    123, 123, 234, 0,
    255, 105, 180, 0,
    79, 129, 189, 0,
    0, 128, 128, 0,
    255, 128, 0, 0,
    //147, 112, 219, 0,
    //40, 160, 120, 0,
    156, 175, 7, 0,
};
*/
#define SIZE_REAL (DIM * DIM)
#define SIZE_COMP (DIM * (DIM / 2 + 1))
int activator_counts[SCALE_COUNT] = {0};
int inhibitor_counts[SCALE_COUNT] = {0};
int variation_counts[SCALE_COUNT] = {0};

float* d_modifiers_values;
int* d_calculation_vector;
unsigned char *d_color_map;
unsigned char *d_tex_data;

float* d_activators;
float* d_inhibitors;
float* d_variations;
int* d_MSTPcolors;
float* d_MSTPvalues;

cufftHandle planR2C;
cufftHandle planC2R;

cufftComplex* d_activators_kernels;
cufftComplex* d_inhibitors_kernels;
cufftComplex* d_variations_kernels;
cufftComplex* d_memory_complex;
cufftComplex* d_MSTPvalues_complex;



void mem_init(){
    cudaMalloc(&d_tex_data, SIZE_REAL * sizeof(unsigned char) * 4);

    cudaMalloc(&d_calculation_vector, SCALE_COUNT * sizeof(int));
    cudaMalloc(&d_modifiers_values, SCALE_COUNT * sizeof(float));
    cudaMalloc(&d_color_map, SCALE_COUNT * sizeof(unsigned char) * 4);
    cudaMalloc(&d_memory_complex, SIZE_COMP * sizeof(cufftComplex));
    cudaMalloc(&d_activators_kernels, SIZE_COMP * SCALE_COUNT * sizeof(cufftComplex));
    cudaMalloc(&d_inhibitors_kernels, SIZE_COMP * SCALE_COUNT * sizeof(cufftComplex));
    cudaMalloc(&d_variations_kernels, SIZE_COMP * SCALE_COUNT * sizeof(cufftComplex));

    cudaMalloc(&d_activators, SIZE_REAL * SCALE_COUNT * sizeof(float));
    cudaMalloc(&d_inhibitors, SIZE_REAL * SCALE_COUNT * sizeof(float));
    cudaMalloc(&d_variations, SIZE_REAL * SCALE_COUNT * sizeof(float));
    cudaMalloc(&d_MSTPvalues_complex, SIZE_COMP * sizeof(cufftComplex));
    cudaMalloc(&d_MSTPcolors, SIZE_REAL * sizeof(unsigned char) * 4);
    cudaMalloc(&d_MSTPvalues, SIZE_REAL * sizeof(float));

    cufftPlan2d(&planC2R, DIM, DIM, CUFFT_C2R);
    cufftPlan2d(&planR2C, DIM, DIM, CUFFT_R2C);

}
void mem_free(){

    cudaFree(d_tex_data);
    cudaFree(d_calculation_vector);
    cudaFree(d_modifiers_values);
    cudaFree(d_color_map);
    cudaFree(d_memory_complex);
    cudaFree(d_activators_kernels);
    cudaFree(d_inhibitors_kernels);
    cudaFree(d_variations_kernels);
    cudaFree(d_activators);
    cudaFree(d_inhibitors);
    cudaFree(d_variations);
    cudaFree(d_MSTPvalues_complex);
    cudaFree(d_MSTPvalues);
    cudaFree(d_MSTPcolors);
    cufftDestroy(planC2R);
    cufftDestroy(planR2C);

}
void val_init(){
    cudaMemcpy(d_calculation_vector, calculation_vector, SCALE_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_modifiers_values, modifiers_values, SCALE_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_color_map, color_map, SCALE_COUNT * sizeof(unsigned char) * 4, cudaMemcpyHostToDevice);
    int block_size = 256;
    dim3 grid(block_size, block_size);
    dim3 block(DIM / block_size, DIM / block_size);
    for(int i = 0; i < SCALE_COUNT; i++){
        init_mat<<<grid, block>>>(d_activators + i * SIZE_REAL, activator_radii[i], DIM, DIM);
        cufftExecR2C(planR2C, d_activators + i * SIZE_REAL, d_activators_kernels + i * SIZE_COMP);
        init_mat<<<grid, block>>>(d_inhibitors + i * SIZE_REAL, inhibitor_radii[i], DIM, DIM);
        cufftExecR2C(planR2C, d_inhibitors + i * SIZE_REAL, d_inhibitors_kernels + i * SIZE_COMP);
        init_mat<<<grid, block>>>(d_variations + i * SIZE_REAL, variation_radii[i], DIM, DIM);
        cufftExecR2C(planR2C, d_variations + i * SIZE_REAL, d_variations_kernels + i * SIZE_COMP);
    }
    init_mat<<<grid, block>>>(d_MSTPvalues, 100, DIM, DIM);
}
void val_compute(){
    int block_size = 256;
    dim3 grid(block_size, block_size);
    dim3 block(DIM / block_size, DIM / block_size);
    cufftExecR2C(planR2C, d_MSTPvalues, d_MSTPvalues_complex);

    for(int s = 0; s < SCALE_COUNT; s++){
        if(calculation_vector[s]){

            multiply_mat<<<grid, block>>>(d_memory_complex, d_MSTPvalues_complex, d_activators_kernels + s * SIZE_COMP, DIM, DIM / 2 + 1);
            cufftExecC2R(planC2R, d_memory_complex, d_activators + s * SIZE_REAL);
            divide_val<<<grid, block>>>(d_activators + s * SIZE_REAL, activator_radii[s] * activator_radii[s], DIM, DIM);

            multiply_mat<<<grid, block>>>(d_memory_complex, d_MSTPvalues_complex, d_inhibitors_kernels + s * SIZE_COMP, DIM, DIM / 2 + 1);
            cufftExecC2R(planC2R, d_memory_complex, d_inhibitors + s * SIZE_REAL);
            divide_val<<<grid, block>>>(d_inhibitors + s * SIZE_REAL, inhibitor_radii[s] * inhibitor_radii[s], DIM, DIM);

            subtract_abs_mat<<<grid, block>>>(d_variations + s * SIZE_REAL, d_inhibitors + s * SIZE_REAL, d_activators + s * SIZE_REAL, DIM, DIM);
            cufftExecR2C(planR2C, d_variations + s * SIZE_REAL, d_memory_complex);
            multiply_mat<<<grid, block>>>(d_memory_complex, d_variations_kernels + s * SIZE_COMP, DIM, DIM / 2 + 1);
            cufftExecC2R(planC2R, d_memory_complex, d_variations + s * SIZE_REAL);
            divide_val<<<grid, block>>>(d_variations + s * SIZE_REAL, variation_radii[s] * variation_radii[s], DIM, DIM);
        }
    }

    calculate_smallest_scale<<<grid, block>>>(d_MSTPcolors, d_variations, d_calculation_vector, DIM, DIM);
    calculate_MSTP<<<grid, block>>>(d_MSTPvalues, d_MSTPcolors, d_activators, d_inhibitors, d_modifiers_values, d_calculation_vector, DIM, DIM);
    float max_val, min_val;
    thrust::pair<float *, float *> result = thrust::minmax_element(thrust::device, d_MSTPvalues, d_MSTPvalues + SIZE_REAL);
    cudaMemcpy(&min_val, result.first, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_val, result.second, sizeof(float), cudaMemcpyDeviceToHost);
    normalize_mat<<<grid, block>>>(d_MSTPvalues, min_val, max_val, DIM, DIM);

    calculate_texture<<<grid, block>>>(d_tex_data, d_color_map, d_MSTPcolors, d_MSTPvalues, DIM, DIM);
}

int main()
{

    mem_init();
    val_init();


    for(int i = 0; i < 1280; i++){
        printf("%d\n", i);
        CUDA_CHECK;
        val_compute();  
    }

    float* h_tex_data = (float*)malloc(sizeof(unsigned char) * DIM * DIM * 4);
    cudaMemcpy(h_tex_data, d_tex_data, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);
    stbi_write_png("image.png", DIM, DIM, 4, h_tex_data, DIM * 4);

    free(h_tex_data);
    mem_free();
    CUDA_CHECK;
    return 0;
}