#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define MAX_WIDTH 2048

__device__ __inline__ void find_linePixels_cuda(int pin1x, int pin1y, int pin2x, int pin2y, int* line_x, int* line_y, int* length, int width) {
    // printf("GPU inline find_linePixels launched successfully!\n");
    
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

__device__ __inline__ size_t l2_norm_cuda_add(unsigned char* constructed_img, unsigned char* inverted_img, int image_size, int width, int* line_x, int* line_y, int line_length, bool isAdd) {
    size_t l2_norm = 0;

    for (int i = 0; i<image_size; i++) {
        int d = constructed_img[i]-inverted_img[i];
        l2_norm += d*d;
    }
    if (isAdd) {
        for (int i = 0; i<line_length; i++) {
            int x = line_x[i];
            int y = line_y[i];
            int new_pixel;
            if (constructed_img[y*width+x] == 0) new_pixel = 200;
            else if (constructed_img[y*width+x] == 255) new_pixel = 255;
            else new_pixel = constructed_img[y*width+x] + 5;
            l2_norm += (new_pixel*new_pixel - 2*new_pixel*inverted_img[y*width+x]
                        - constructed_img[y*width+x]*constructed_img[y*width+x] + 
                        + 2*constructed_img[y*width+x]*inverted_img[y*width+x]);
        }
    }
    else {
        for (int i = 0; i<line_length; i++) {
            int x = line_x[i];
            int y = line_y[i];
            l2_norm +=  - constructed_img[y*width+x]*constructed_img[y*width+x] + 
                        + 2*constructed_img[y*width+x]*inverted_img[y*width+x];
        }
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
   
    // printf("kernel launched [%d, %d] (%d, %d) (%d, %d) \n", i, j, x_coords[i], y_coords[i], x_coords[j], y_coords[j]);
    // printf("launching find_linePixels\n");
    find_linePixels_cuda(x_coords[i], y_coords[i], x_coords[j], y_coords[j], line_x, line_y, &line_length, cropped_width);
    // if (i == 0 && j == 0){
    //     printf("line length %d\n", line_length);
    //     for (int index = 0; index < cropped_width; index++){
    //         printf("%d ", line_x[index]);
    //     }
    //     printf("\n");
    // }
    // printf("launching l2 norm\n");
    tmp_norm = l2_norm_cuda_add(constructed_img, inverted_img, cropped_size, cropped_width, line_x, line_y, line_length, true);
    // printf("tmp_norm %d\n", (int)tmp_norm);
    // if (i == 49 && j == 15) printf("(49,15) %d\n", (int)tmp_norm);

    // FIXME: critical section here is
    // if (tmp_norm < *bestNorm && i > j) {
    //     *bestPin1 = i;
    //     *bestPin2 = j;
    //     *bestNorm = tmp_norm;
    // }
    bestNorm[i*numPins + j] = tmp_norm;
    // if (i == 63 && j == 45) printf("!!(63,45) %u", tmp_norm);
    
    __syncthreads();
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
    // cudaMemcpy(device_bestPin1, bestPin1, pin_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_bestPin2, bestPin2, pin_size, cudaMemcpyHostToDevice);

    int norm_size = sizeof(size_t)*numPins*numPins;
    size_t* device_bestNorm;
    cudaMalloc(&device_bestNorm, norm_size);
    // cudaMemcpy(device_bestNorm, bestNorm, norm_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((numPins + blockDim.x - 1) / blockDim.x,
                 (numPins + blockDim.y - 1) / blockDim.y);
    size_t memorysize = 2 * MAX_WIDTH * sizeof(int);
    find_best_pins_kernel<<<gridDim, blockDim, memorysize>>>(device_x_coords, device_y_coords, numPins, cropped_width,
        device_bestNorm,
        device_constructed_img, device_inverted_img, cropped_size);
    
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    cudaDeviceSynchronize(); 

    size_t* norm = (size_t*)malloc(sizeof(size_t)*numPins*numPins);
    
    cudaMemcpy(norm, device_bestNorm, norm_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numPins; i++){
        for (int j = 0; j < i; j++){
            size_t tmp_norm = norm[i*numPins+j];
            if (tmp_norm == 1010762398) printf("!!!1010762398 (%d, %d) ", i, j);
            // printf("(%d|%d, %d|%d) %u \n", i, pin1[i*numPins+j], j, pin2[i*numPins+j], tmp_norm);
            // if (i == 63 && j == 45) printf("!!(63,45) %u", tmp_norm);
            if (tmp_norm < *bestNorm) {
                // noRemoval = false;
                *bestPin1 = i;
                *bestPin2 = j;
                *bestNorm = tmp_norm;
            }
        }
    }

    // printf("(%d, %d) %d\n", *bestPin1, *bestPin2, *bestNorm);
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


// extern void find_linePixels(int pin1x, int pin1y, int pin2x, int pin2y, int* line_x, int* line_y, int* length, int width);
// extern size_t l2_norm(unsigned char* constructed_img, unsigned char* inverted_img, int image_size, int width, int* line_x, int* line_y, int line_length, bool isAdd);

// int line_length;
// int line_x[width];
// int line_y[width];
// for (size_t i = 0; i<numPins; i++) {

//     for (size_t j = 0; j<i; j++) {
//         // printf("cp1, line_length: %d\n", line_length);
//         find_linePixels(x_coords[i], y_coords[i], x_coords[j], y_coords[j], line_x, line_y, &line_length, cropped_width);
//         // printf("cp2, line_length: %d\n", line_length);
//         size_t tmp_norm = l2_norm(constructed_img, inverted_img, cropped_size, cropped_width, line_x, line_y, line_length, true);
//         // printf("i: %lu,j: %lu, tmp_norm: %lu, bestNorm: %lu\n", i, j, tmp_norm, bestNorm);
//         if (tmp_norm < bestNorm) {
//             noRemoval = false;
//             // printf("tmp_norm < bestNorm\n");
//             bestPin1 = i;
//             bestPin2 = j;
//             bestNorm = tmp_norm;
//         }
//         // printf("cp3\n");
//     }
// }

// __global__ void reduceNorm(int numPins, 
//     volatile unsigned char* constructed_img, volatile unsigned char* inverted_img, 
//     int image_size, int width, int pinsPairNum, size_t* device_norms,
//     volatile int* x_coords, volatile int* y_coords, bool isAdd){
//     int line_length;
//     int line_x[width];
//     int line_y[width];

//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     // find the pin pairs using the index
//     // TODO: this prob can be found more efficient
//     int g = pinsPairNum;
//     int pin1 = pinsPairNum;
//     while (g > index) {
//         g -= (1+pin1);
//         pin1 -= 1;
//     }
//     pin2 = index - g;
//     find_linePixels(x_coords[pin1], y_coords[pin1], x_coords[pin2], y_coords[pin2], line_x, line_y, &line_length, cropped_width);
//     device_norms[index] = l2_norm(constructed_img, inverted_img, cropped_size, cropped_width, line_x, line_y, line_length, true);
//     __syncthreads();

//     // reducing to find max reduction
//     const int tid = threadIdx.x;

//     auto step_size = 1;
//     int number_of_threads = blockDim.x;

//     while (number_of_threads > 0)
//     {
//         if (tid < number_of_threads) // still alive?
//         {
//             const auto fst = tid * step_size * 2;
//             const auto snd = fst + step_size;
//             if (device_norms[snd] < device_norms[fst]) device_norms[fst] = device_norms[snd];
//         }

//         step_size <<= 1; 
//         number_of_threads >>= 1;
//     }

// }
// void findMaxNormReduceLineCuda(int numPins, 
//     unsigned char* constructed_img, unsigned char* inverted_img, 
//     int image_size, int width, 
//     int* x_coords, int* y_coords, bool isAdd) {

//     int pinsPairNum = 0; // number of distinct pin pairs
//     for (size_t i = 0; i<numPins; i++) {
//         pinsPairNum += i;
//     }

//     const int threadsPerBlock = 512;
//     const int blocks = (pinsPairNum + threadsPerBlock - 1) / threadsPerBlock;

//     unsigned char* device_constructed_img;
//     unsigned char* device_inverted_img;
//     int* device_x_coords;
//     int* device_y_coords;
//     size_t* device_norms;

//     cudaMalloc(&device_constructed_img, sizeof(unsigned char) * image_size);
//     cudaMalloc(&device_inverted_img, sizeof(unsigned char) * image_size);
//     cudaMalloc(&device_x_coords, sizeof(int) * numPins);
//     cudaMalloc(&device_y_coords, sizeof(int) * numPins);
//     cudaMalloc(&device_norms, sizeof(size_t) * pinsPairNum);

//     cudaMemcpy(device_constructed_img, constructed_img, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_inverted_img, inverted_img, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_x_coords, x_coords, sizeof(int) * numPins, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_y_coords, y_coords, sizeof(int) * numPins, cudaMemcpyHostToDevice);

//     reduceNorm(numPins, device_constructed_img, device_inverted_img, 
//         image_size, width, pinsPairNum, device_norms,
//         device_x_coords, device_y_coords, isAdd);

//     size_t result;
//     cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);

// }
