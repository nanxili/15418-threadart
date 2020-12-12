#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define MAX_WIDTH 2048

extern void draw_all_lines(int* found_p1, int* found_p2, int line_count, int* x_coords, int* y_coords, unsigned char* constructed_img, unsigned char* img, int width);

__device__ __inline__ void find_linePixels_cuda(int pin1x, int pin1y, int pin2x, int pin2y, int* line_x, int* line_y, int* length, int width) {
    if (pin1x == pin2x) { // if same x coords, draw vertical line
        int startY = pin1y>pin2y ? pin2y : pin1y;
        int endY = pin1y>pin2y ? pin1y : pin2y;
        for (size_t i = startY; i<endY; i++) {
            line_x[i-startY] = pin1x;
            line_y[i-startY] = i;
        }
        *length = endY - startY;
        return;
    }

    if (pin1y == pin2y) { // if same x coords, draw vertical line
        int startX = pin1x>pin2x ? pin2x : pin1x;
        int endX = pin1x>pin2x ? pin1x : pin2x;
        for (size_t i = startX; i<endX; i++) {
            line_y[i-startX] = pin1y;
            line_x[i-startX] = i;
        }
        *length = endX - startX;
        return;
    }
    // find the line that passes through two points
    float a, b;
    int startX, endX, startY, endY;
    if (abs(pin1x-pin2x) > abs(pin1y-pin2y)) {
        if (pin1x>pin2x) {
            startX = pin2x;
            startY = pin2y;
            endX = pin1x;
            a = ((float)pin1y)-pin2y;
            b = ((float)pin1x)-pin2x;
        }
        else {
            startX = pin1x;
            startY = pin1y;
            endX = pin2x;
            a = ((float)pin2y)-pin1y;
            b = ((float)pin2x)-pin1x;
        }
        float m = a/b;
        float k = startY - (m * startX);
        for (size_t i = startX; i<endX; i++) {
            line_x[i-startX] = i;
            float y = m*i+k;
            if (y>=(width*1.0)) y = ((width-1)*1.0);
            if (y<0) y = 0.0;
            line_y[i-startX] = floor(y);
        }
        *length = endX-startX;
    }
    else {
        if (pin1y>pin2y) {
            startX = pin2x;
            startY = pin2y;
            endY = pin1y;
            a = ((float)pin1y)-pin2y;
            b = ((float)pin1x)-pin2x;
        }
        else {
            startX = pin1x;
            startY = pin1y;
            endY = pin2y;
            a = ((float)pin2y)-pin1y;
            b = ((float)pin2x)-pin1x;
        }
        float m = a/b;
        float k = startY - (m * startX);
        for (size_t i = startY; i<endY; i++) {
            line_y[i-startY] = i;
            float x = (i-k)/m;
            if (x>=(width*1.0)) x = ((width-1)*1.0);
            if (x<0) x = 0.0;
            line_x[i-startY] = floor(x);
        }
        *length = endY-startY;
    }
    
}

__device__ __inline__ size_t l2_norm_cuda_add(unsigned char* constructed_img, unsigned char* inverted_img, int image_size, int width, int* line_x, int* line_y, int line_length) {
    size_t l2_norm = 0;

    for (int i = 0; i<image_size; i++) {
        int d = constructed_img[i]-inverted_img[i];
        l2_norm += d*d;
    }
    for (int i = 0; i<line_length; i++) {
        int x = line_x[i];
        int y = line_y[i];
        int new_pixel;
        if (constructed_img[y*width+x] == 0) new_pixel = 100;
        else if (constructed_img[y*width+x] == 255) new_pixel = 255;
        else new_pixel = constructed_img[y*width+x] + 5;
        l2_norm += (new_pixel*new_pixel - 2*new_pixel*inverted_img[y*width+x]
                    - constructed_img[y*width+x]*constructed_img[y*width+x] + 
                    + 2*constructed_img[y*width+x]*inverted_img[y*width+x]);
    }
    
    return l2_norm;
}

__device__ __inline__ size_t l2_norm_cuda_sub(unsigned char* constructed_img, unsigned char* inverted_img, int image_size, int width, int* line_x, int* line_y, int line_length) {
    size_t l2_norm = 0;

    for (int i = 0; i<image_size; i++) {
        int d = constructed_img[i]-inverted_img[i];
        l2_norm += d*d;
    }
    for (int i = 0; i<line_length; i++) {
        int x = line_x[i];
        int y = line_y[i];
        int new_pixel;
        if (constructed_img[y*width+x] == 0) new_pixel = 0;
        else if (constructed_img[y*width+x] == 100) new_pixel = 0;
        else new_pixel = constructed_img[y*width+x] - 5;
        l2_norm += (new_pixel*new_pixel - 2*new_pixel*inverted_img[y*width+x]
                    - constructed_img[y*width+x]*constructed_img[y*width+x] + 
                    + 2*constructed_img[y*width+x]*inverted_img[y*width+x]);
    }
    
    return l2_norm;
}

__global__ void find_best_pins_kernel(int* x_coords, int* y_coords, int numPins, int cropped_width, 
    size_t* bestNorm,
    unsigned char* constructed_img, unsigned char* inverted_img, int cropped_size){

    int line_x[MAX_WIDTH];
    int line_y[MAX_WIDTH];
    int line_length;
    size_t tmp_norm = 0;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
   
    find_linePixels_cuda(x_coords[i], y_coords[i], x_coords[j], y_coords[j], line_x, line_y, &line_length, cropped_width);
    tmp_norm = l2_norm_cuda_add(constructed_img, inverted_img, cropped_size, cropped_width, line_x, line_y, line_length);
    bestNorm[i*numPins + j] = tmp_norm;
}

void find_best_pins(int* x_coords, int* y_coords, int numPins, int cropped_width, 
    int* bestPin1, int* bestPin2, size_t* bestNorm,
    unsigned char* constructed_img, unsigned char* inverted_img, int cropped_size){
    
    int coords_size = sizeof(int) * numPins;
    int* device_x_coords;
    int* device_y_coords;
    cudaMalloc(&device_x_coords, coords_size);
    cudaMalloc(&device_y_coords, coords_size);
    cudaMemcpy(device_x_coords, x_coords, coords_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y_coords, y_coords, coords_size, cudaMemcpyHostToDevice);
    
    int img_size = sizeof(unsigned char) * cropped_size;
    unsigned char* device_constructed_img;
    unsigned char* device_inverted_img;
    cudaMalloc(&device_constructed_img, img_size);
    cudaMalloc(&device_inverted_img, img_size);
    cudaMemcpy(device_constructed_img, constructed_img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_inverted_img, inverted_img, img_size, cudaMemcpyHostToDevice);
    
    int pin_size = sizeof(int)*numPins*numPins;
    int* device_bestPin1;
    int* device_bestPin2;
    cudaMalloc(&device_bestPin1, pin_size);
    cudaMalloc(&device_bestPin2, pin_size);

    int norm_size = sizeof(size_t)*numPins*numPins;
    size_t* device_bestNorm;
    cudaMalloc(&device_bestNorm, norm_size);

    dim3 blockDim(16,16);
    dim3 gridDim((numPins + blockDim.x - 1) / blockDim.x,
                 (numPins + blockDim.y - 1) / blockDim.y);
    size_t memorysize = 2 * MAX_WIDTH * sizeof(int);
    find_best_pins_kernel<<<gridDim, blockDim, memorysize>>>(device_x_coords, device_y_coords, numPins, cropped_width,
        device_bestNorm,
        device_constructed_img, device_inverted_img, cropped_size);
    
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        // fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        printf("in add WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize(); 

    size_t* norm = (size_t*)malloc(sizeof(size_t)*numPins*numPins);
    
    cudaMemcpy(norm, device_bestNorm, norm_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numPins; i++){
        for (int j = 0; j < i; j++){
            size_t tmp_norm = norm[i*numPins+j];
            if (tmp_norm < *bestNorm) {
                *bestPin1 = i;
                *bestPin2 = j;
                *bestNorm = tmp_norm;
            }
        }
    }

    free(norm);
    cudaFree(device_x_coords);
    cudaFree(device_y_coords);
    cudaFree(device_constructed_img);
    cudaFree(device_inverted_img);
    cudaFree(device_bestPin1);
    cudaFree(device_bestPin2);
    cudaFree(device_bestNorm);
    return;
}

__global__ void remove_lines_kernel(
    int* x_coords, int* y_coords, int numPins, int cropped_width, size_t bestNorm,
    int* found_pin1, int* found_pin2, int line_count,
    unsigned char* constructed_img, unsigned char* inverted_img, int cropped_size,
    size_t* new_norms){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= line_count) return;
    int p1 = found_pin1[i];
    int p2 = found_pin2[i];
    if (p1 == 0 && p2 == 0){
        new_norms[i] = 0;
        return;
    }
  
    int line_x[MAX_WIDTH];
    int line_y[MAX_WIDTH];
    int line_length;
    size_t tmp_norm = 0;
   
    find_linePixels_cuda(x_coords[p1], y_coords[p1], x_coords[p2], y_coords[p2], line_x, line_y, &line_length, cropped_width);
    tmp_norm = l2_norm_cuda_sub(constructed_img, inverted_img, cropped_size, cropped_width, line_x, line_y, line_length);
    if (tmp_norm < bestNorm) {
        new_norms[i] = tmp_norm;
    }
    else {
        new_norms[i] = 0;
    }
}
  
int remove_lines(
int* x_coords, int* y_coords, int numPins, int cropped_width, size_t* bestNorm,
int* found_pin1, int* found_pin2, int line_count,
unsigned char* constructed_img, unsigned char* inverted_img, unsigned char* img, int cropped_size){

    int coords_size = sizeof(int) * numPins;
    int* device_x_coords;
    int* device_y_coords;
    cudaMalloc(&device_x_coords, coords_size);
    cudaMalloc(&device_y_coords, coords_size);
    cudaMemcpy(device_x_coords, x_coords, coords_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y_coords, y_coords, coords_size, cudaMemcpyHostToDevice);

    int img_size = sizeof(unsigned char) * cropped_size;
    unsigned char* device_constructed_img;
    unsigned char* device_inverted_img;
    cudaMalloc(&device_constructed_img, img_size);
    cudaMalloc(&device_inverted_img, img_size);
    cudaMemcpy(device_constructed_img, constructed_img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_inverted_img, inverted_img, img_size, cudaMemcpyHostToDevice);

    int lines_size = sizeof(int) * line_count;
    int* device_found_pin1;
    int* device_found_pin2;
    cudaMalloc(&device_found_pin1, lines_size);
    cudaMalloc(&device_found_pin2, lines_size);
    cudaMemcpy(device_found_pin1, found_pin1, lines_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_found_pin2, found_pin2, lines_size, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    int gridNum = (line_count + blockDim.x - 1) / blockDim.x;
    dim3 gridDim(gridNum);

    int threadNum = gridNum*256;
    size_t new_norms[threadNum];
    int new_norms_size = sizeof(size_t)*threadNum;
    size_t* device_new_norms;
    cudaMalloc(&device_new_norms, new_norms_size);
    cudaMemcpy(device_new_norms, new_norms, new_norms_size, cudaMemcpyHostToDevice);

    size_t new_bestNorm = 0;
    size_t local_bestNorm = *bestNorm;
    int remove_line_count = 0;
    bool removed_line = true;

    while (removed_line){
        removed_line = false;
        if (new_bestNorm != 0) local_bestNorm = new_bestNorm;
        new_bestNorm = 0;
        remove_lines_kernel<<<gridDim, blockDim>>>(device_x_coords, device_y_coords, numPins, cropped_width, local_bestNorm,
        device_found_pin1, device_found_pin2, line_count,
        device_constructed_img, device_inverted_img, cropped_size,
        device_new_norms);
        cudaError_t errCode = cudaPeekAtLastError();
        if (errCode != cudaSuccess) {
            // fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
            printf("in sub WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }
        cudaDeviceSynchronize(); 

        cudaMemcpy(new_norms, device_new_norms, new_norms_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < line_count; i++){
            if (new_norms[i] != 0){
                new_bestNorm = new_norms[i];
                printf("removing (%d, %d)\n", found_pin1[i], found_pin2[i]);
                found_pin1[i] = 0;
                found_pin2[i] = 0;
                remove_line_count ++;
                removed_line = true;
                break;
            }
        }
        draw_all_lines(found_pin1, found_pin2, line_count, x_coords, y_coords, constructed_img, img, cropped_width);
        cudaMemcpy(device_found_pin1, found_pin1, lines_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_found_pin2, found_pin2, lines_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_constructed_img, constructed_img, img_size, cudaMemcpyHostToDevice);
    }
    *bestNorm = local_bestNorm;

    cudaFree(device_x_coords);
    cudaFree(device_y_coords);
    cudaFree(device_constructed_img);
    cudaFree(device_inverted_img);
    cudaFree(device_found_pin1);
    cudaFree(device_found_pin2);
    cudaFree(device_new_norms);

    return remove_line_count;
}



