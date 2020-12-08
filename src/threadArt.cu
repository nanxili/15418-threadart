#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>


__global__ void kernel() {
    printf("Hello world from GPU!\n");
}
void hello_world() {
    printf("Hello world!\n");
    kernel<<<1,1>>>();
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
