#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "kernels.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int activator_radii[SCALE_COUNT]    = {4,8,16,32,64,128,256,512,1024};
int inhibitor_radii[SCALE_COUNT]    = {8,16,32,64,128,256,512,1024,2048};
int variation_radii[SCALE_COUNT]    = {1,1,1,1,1,1,1,1,1};
float modifiers_values[SCALE_COUNT] = {0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002};
int calculation_vector[SCALE_COUNT] = {0,0,0,0,0,0,0,0};
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

/* those look nice*/
/*
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
};
*/
unsigned char color_map[SCALE_COUNT * 4] = {
    220, 70, 89, 0,
    123, 123, 234, 0,
    255, 105, 180, 0,
    79, 129, 189, 0,
    0, 128, 128, 0,
    255, 128, 0, 0,
    147, 112, 219, 0,
    40, 160, 120, 0,
    156, 175, 7, 0,
};
#define SIZE_REAL (DIM * DIM)
#define SIZE_COMP (DIM * (DIM / 2 + 1))
int activator_counts[SCALE_COUNT] = {0};
int inhibitor_counts[SCALE_COUNT] = {0};
int variation_counts[SCALE_COUNT] = {0};

float* d_modifiers_values;
int* d_calculation_vector;
unsigned char *d_color_map;
unsigned char *d_tex_data;
GLuint tex_ID;
cudaGraphicsResource_t tex_res;

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
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        unsigned char* h_data;
        unsigned char* h_data_big;
        switch (key)
        {
            case GLFW_KEY_1:
                calculation_vector[0] = abs(calculation_vector[0] - 1);
                break;
            case GLFW_KEY_2:
                calculation_vector[1] = abs(calculation_vector[1] - 1);
                break;
            case GLFW_KEY_3:
                calculation_vector[2] = abs(calculation_vector[2] - 1);
                break;
            case GLFW_KEY_4:
                calculation_vector[3] = abs(calculation_vector[3] - 1);
                break;
            case GLFW_KEY_Q:
                calculation_vector[4] = abs(calculation_vector[4] - 1);
                break;
            case GLFW_KEY_W:
                calculation_vector[5] = abs(calculation_vector[5] - 1);
                break;
            case GLFW_KEY_E:
                calculation_vector[6] = abs(calculation_vector[6] - 1);
                break;
            case GLFW_KEY_R:
                calculation_vector[7] = abs(calculation_vector[7] - 1);
                break;
            case GLFW_KEY_A:
                calculation_vector[8] = abs(calculation_vector[8] - 1);
                break;
            case GLFW_KEY_S:
                calculation_vector[9] = abs(calculation_vector[9] - 1);
                break;
            case GLFW_KEY_D:
                calculation_vector[10] = abs(calculation_vector[10] - 1);
                break;
            case GLFW_KEY_F:
                calculation_vector[11] = abs(calculation_vector[11] - 1);
                break;
            case GLFW_KEY_Z:
                calculation_vector[12] = abs(calculation_vector[12] - 1);
                break;
            case GLFW_KEY_X:
                calculation_vector[13] = abs(calculation_vector[13] - 1);
                break;
            case GLFW_KEY_C:
                calculation_vector[14] = abs(calculation_vector[14] - 1);
                break;
            case GLFW_KEY_V:
                calculation_vector[15] = abs(calculation_vector[15] - 1);
                break;
            case GLFW_KEY_SPACE:
                memset(calculation_vector, 0, sizeof(int) * SCALE_COUNT);
                break;
            case GLFW_KEY_F1:
                h_data = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4);
                cudaGraphicsMapResources(1, &tex_res, 0);
                cudaGraphicsResourceGetMappedPointer((void **)&d_tex_data, NULL, tex_res);
                cudaMemcpy(h_data, d_tex_data, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);
                cudaGraphicsUnmapResources(1, &tex_res, 0);
                stbi_write_png("image.png", DIM, DIM, 4, h_data, DIM * 4);
                free(h_data);
                break;
            case GLFW_KEY_F2:
                h_data = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4);
                cudaGraphicsMapResources(1, &tex_res, 0);
                cudaGraphicsResourceGetMappedPointer((void **)&d_tex_data, NULL, tex_res);
                cudaMemcpy(h_data, d_tex_data, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);
                cudaGraphicsUnmapResources(1, &tex_res, 0);
                h_data_big = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4 * 4);
                size_t stride = DIM * 4 * sizeof(unsigned char);
                for(int i = 0; i < DIM; i++){
                    memcpy(h_data_big + i * stride * 2,                                 h_data + i * stride, stride);
                    memcpy(h_data_big + i * stride * 2 + stride,                        h_data + i * stride, stride);
                    memcpy(h_data_big + stride * 2 * DIM + i * stride * 2,              h_data + i * stride, stride);
                    memcpy(h_data_big + stride * 2 * DIM + i * stride * 2 + stride,     h_data + i * stride, stride);
                }
                stbi_write_png("image.png", DIM * 2, DIM * 2, 4, h_data_big, DIM * 4 * 2);
                free(h_data_big);
                free(h_data);
                break;
        }
        cudaMemcpy(d_calculation_vector, calculation_vector, SCALE_COUNT * sizeof(int), cudaMemcpyHostToDevice);
        for(int i = 0; i < SCALE_COUNT; i++){
            if(i % 4 == 0) printf("\n");
            printf("%d\t",calculation_vector[i]);
        }
        printf("\n");
    }
}
int main()
{
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "MSTP", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glewInit();

    glGenBuffers(1, &tex_ID);
    glBindBuffer(GL_ARRAY_BUFFER, tex_ID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned char) * DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&tex_res, tex_ID, cudaGraphicsMapFlagsNone);

    glGenTextures(1, &tex_ID);
    glBindTexture(GL_TEXTURE_2D, tex_ID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, DIM, DIM, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    mem_init();
    val_init();

    int i = 0;
    while (!glfwWindowShouldClose(window))
    {	
        if(i % 30 == 0) printf("%d\n", i);
        i++;
        glfwPollEvents();
		cudaDeviceSynchronize();

        cudaGraphicsMapResources(1, &tex_res, 0);

        cudaGraphicsResourceGetMappedPointer((void **)&d_tex_data, NULL, tex_res);

        val_compute();  
        //cudaMemcpy(d_tex_data, d_MSTPvalues, DIM * DIM * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &tex_res, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, tex_ID);

        glBindTexture(GL_TEXTURE_2D, tex_ID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glBindTexture(GL_TEXTURE_2D, tex_ID);
        glEnable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);


        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1.0, -1.0, 0.5);
        glTexCoord2f(1.0, 0.0);
        glVertex3f(1.0, -1.0, 0.5);
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1.0, 1.0, 0.5);
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1.0, 1.0, 0.5);
        glEnd();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();

        glDisable(GL_TEXTURE_2D);


        glfwSwapBuffers(window);
    }

    mem_free();
    glDeleteTextures(1, &tex_ID);
    glfwTerminate();

    return 0;
}