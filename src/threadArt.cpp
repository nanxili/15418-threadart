#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <limits> 
#include <queue>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image/stb_image_resize.h"

std::clock_t currTime;
double duration;

#define BLACK 0

#define TIMER(prevTime, text) \
    currTime = std::clock(); \
    duration = ( currTime - prevTime ) / (double) CLOCKS_PER_SEC; \
    prevTime = currTime; \
    std::cout << text << " takes " << duration << "s" << std::endl; \

size_t truncate(int32_t value)
{   
    if(value < 0) return 0;
    if(value > 255) return 255;

    return value;
}

unsigned char* read_image(int* width, int* height, int* channels) {
    unsigned char* img;
    img = stbi_load("../test_images/krisWu.jpg", width, height, channels, 0);
    if(img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", *width, *height, *channels);
    return img;
}

void gray_scale_image(unsigned char *orig_img, int img_size, unsigned char *gray_img, int channels, int gray_channels) {
    if(gray_img == NULL) {
        printf("Unable to allocate memory for the gray image.\n");
        exit(1);
    }
    for(unsigned char *p = orig_img, *pg = gray_img; p != orig_img + img_size; p += channels, pg += gray_channels) {
        *pg = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
        if(channels == 4) {
            *(pg + 1) = *(p + 3);
        }
    }
    printf("Converted to grayscale\n");
}

void contrast_image(unsigned char *img, int img_size, int channels) {
    int contrast = -128;
    float factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));
    for(unsigned char *pg = img; pg != img + img_size; pg += channels) {
        uint8_t p = (uint8_t)*pg;
        int32_t np = (int32_t)(uint32_t)p;
        *pg = (uint8_t)truncate((factor * (np - 128) + 128));
    }
    printf("Applied contrast\n");
}

void crop_circle(unsigned char * img, int width, int radius) {
    int x0 = radius;
    int y0 = radius;
    for (size_t i=0; i<width; i++) {
        for (size_t j=0; j<width; j++) {
            int x = x0-i;
            int y = y0-j;
            if ((x*x + y*y) > radius*radius) {
                img[j*width+i] = 255;
            }
        }
    }
    printf("Cropped circle\n");
}
void invert_image(unsigned char *img, unsigned char *inverted_img, int img_size, int channels) {
    for(unsigned char *pg = img, *i_pg = inverted_img; pg != img + img_size; pg += channels, i_pg += channels) {
        *i_pg = 255-(uint8_t)*pg;
    }
    printf("Inverted image\n");
}

void find_pinCords(int numPins, int radius, int width, int* x_coords, int* y_coords) {
    // pins are found in order (radius,0) -> (0, -radius) -> (-radius, 0) -> (9, radius) -> (radius,0)
    int x0 = radius;
    int y0 = radius;
    float a = 2*M_PI/numPins;
    for (size_t i=0; i<numPins; i++) {
        float angle = a * i;
        float x = x0 + radius*cos(angle);
        float y = y0 + radius*sin(angle);
        x_coords[i] = (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
        if (x_coords[i] >= width) x_coords[i] = width-1;
        if (x_coords[i] <0) x_coords[i] = 0;
        y_coords[i] = (y > 0.0) ? floor(y + 0.5) : ceil(y - 0.5);
        if (y_coords[i] >= width) y_coords[i] = width-1;
        if (y_coords[i] <0) y_coords[i] = 0;
        
    }
    printf("Found pinCords\n");
}

void plot_pinCords(unsigned char *img, int numPins, int width, int* x_coords, int* y_coords) {
    // the greater the i, the darker the pin
    // printf("inside plot_pinCords, width: %d, numPins: %d \n", width, numPins);
    for (size_t i=0; i<numPins; i++) {
        int x = x_coords[i];
        int y = y_coords[i];
        // printf("i: %lu, x: %d, y: %d \n", i, x, y);
        // make pin 9 pixels size to be more visible
        if (y>0) img[(y-1)*width+x] = 255-(255*i/numPins);
        if (y>1) img[(y-2)*width+x] = 255-(255*i/numPins);
        if (y<width-1) img[(y+1)*width+x] = 255-(255*i/numPins);
        if (y<width-2) img[(y+2)*width+x] = 255-(255*i/numPins);
        if (x>0) img[y*width+x-1] = 255-(255*i/numPins);
        if (x>1) img[y*width+x-2] = 255-(255*i/numPins);
        if (x<width-1) img[y*width+x+1] = 255-(255*i/numPins);
        if (x<width-2) img[y*width+x+2] = 255-(255*i/numPins);
        img[y*width+x] = 255-(255*i/numPins);
    }
    printf("Plotted pinCords\n");
}

void find_linePixels(int pin1x, int pin1y, int pin2x, int pin2y, int* line_x, int* line_y, int* length, int width) {
    // printf("find_linePixels pin1x: %d, pin1y: %d, pin2x: %d, pin2y: %d\n", pin1x, pin1y, pin2x, pin2y);
    if (pin1x == pin2x) { // if same x coords, draw vertical line
        // printf("pin1x == pin2x\n");
        int startY = pin1y>pin2y ? pin2y : pin1y;
        int endY = pin1y>pin2y ? pin1y : pin2y;
        for (size_t i = startY; i<endY; i++) {
            line_x[i-startY] = pin1x;
            line_y[i-startY] = i;
        }
        *length = endY - startY;
        return;
    }
    // find the line that passes through two points
    float a, b, c;
    int startX, endX, startY;
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
    // printf("startX: %d, endX: %d, a: %f, c: %f\n", startX, endX, a, c);
    for (size_t i = startX; i<endX; i++) {
        line_x[i-startX] = i;
        float y = m*i+k;
        if (y>=(width*1.0)) y = ((width-1)*1.0);
        if (y<0) y = 0.0;
        line_y[i-startX] = floor(y);
        // if(line_y[i-startX]>=width) printf("width: %d, line_y[i-startX]<width: %d\n", width, line_y[i-startX]);
        assert(line_y[i-startX]<width);
    }
    *length = endX-startX;
}

void drawLine(unsigned char *img, int* x_coords, int* y_coords, int length, int width) {
    // printf("length = %d \n", length);
    for (size_t i = 0; i<length; i++) {
        int x = x_coords[i];
        int y = y_coords[i];
        img[y*width+x] = BLACK;
    }
}

void undoDrawLine(unsigned char *img, unsigned char* inverted_img, int* x_coords, int* y_coords, int length, int width) {
    // printf("length = %d \n", length);
    for (size_t i = 0; i<length; i++) {
        int x = x_coords[i];
        int y = y_coords[i];
        img[y*width+x] = 255-inverted_img[y*width+x];
    }
}

size_t l2_norm(unsigned char* constructed_img, unsigned char* inverted_img, int image_size, int width, int* line_x, int* line_y, int line_length, bool isAdd) {
    unsigned char tmp_img[image_size];
    size_t l2_norm = 0;
    for (size_t i = 0; i<image_size; i++) {
        tmp_img[i] = constructed_img[i];
    }
    // printf("cp0.1 line_length: %d\n", line_length);

    for (size_t i = 0; i<line_length; i++) {
        // printf("i: %lu, p0.1.1\n", i);
        int x = line_x[i];
        int y = line_y[i];
        // printf("x: %d, y: %d, p0.1.2\n", x, y);
        if (isAdd) tmp_img[y*width+x] = 255;
        else tmp_img[y*width+x] = 0;
    }
    // printf("cp0.2\n");

    for (size_t i = 0; i<image_size; i++) {
        // printf("tmp_img[i]: %d, inverted_img[i]: %d \n", (uint8_t)tmp_img[i], (uint8_t)inverted_img[i]);
        size_t d = tmp_img[i]-inverted_img[i];
        // if (d!=0) printf("tmp_img[i]: %d, inverted_img[i]: %d \n", (uint8_t)tmp_img[i], (uint8_t)inverted_img[i]);
        l2_norm += d*d;
        // if(d!=0) printf("l2_norm: %lu \n", l2_norm);
    }
    // printf("cp0.3\n");
    return l2_norm;
}

void add_line2Img(unsigned char* constructed_img, unsigned char* img, int width, int* line_x, int* line_y, int length) {
    for (size_t i = 0; i<length; i++) {
        int x = line_x[i];
        int y = line_y[i];
        constructed_img[y*width+x] = 255;
    }
    drawLine(img, line_x, line_y, length, width);
}

void remove_lineFromImg(unsigned char* constructed_img, unsigned char* inverted_img, int width, int* line_x, int* line_y, int length) {
    for (size_t i = 0; i<length; i++) {
        int x = line_x[i];
        int y = line_y[i];
        constructed_img[y*width+x] = 0;
    }
    undoDrawLine(constructed_img, inverted_img,line_x, line_y, length, width);
}

int main(void) {
    std::clock_t prevTime;
    prevTime = std::clock();

    const int numPins = 64;
    const int numLines = 10;
    const float threadThickness = 0.15;
    const float frameDiameter = 614.4;

    int width, height, channels;
    // read input image
    unsigned char* input_img = read_image(&width, &height, &channels);
    TIMER(prevTime, "reading image")

    // Convert the input image to gray
    int gray_channels = channels == 4 ? 2 : 1;
    size_t img_size = width * height * channels;
    size_t gray_img_size = width * height * gray_channels;
    unsigned char *gray_img = (unsigned char*)malloc(gray_img_size);
    gray_scale_image(input_img, img_size, gray_img, channels, gray_channels);
    stbi_write_jpg("../test_images/krisWu_gray.jpg", width, height, gray_channels, gray_img, 100);
    
    // crop image to a square
    size_t shorterEdge = width>height ? height : width;
    float widthFrac = 1-((float)shorterEdge)/width;
    float heightFrac = 1-((float)shorterEdge)/height;
    size_t cropped_width = 1; while (cropped_width <= shorterEdge) cropped_width <<= 1; cropped_width >>= 1;
    cropped_width /= 2;
    printf("shorterEdge: %lu, cropped_width: %lu \n", shorterEdge, cropped_width);
    size_t cropped_size = cropped_width * cropped_width * gray_channels;
    unsigned char* img = (unsigned char*)malloc(cropped_size);
    stbir_resize_region(gray_img, width, height, 0, img, cropped_width, cropped_width, 0, 
        STBIR_TYPE_UINT8, gray_channels, -1, 0, 
        STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR, 
        NULL, widthFrac/2, heightFrac/2, 1-widthFrac/2, 1-heightFrac/2);
    stbi_write_jpg("../test_images/krisWu_cropped.jpg", cropped_width, cropped_width, gray_channels, img, 100);
    free(input_img);
    free(gray_img);

    // adds contrast to image
    contrast_image(img, cropped_size, gray_channels); 
    stbi_write_jpg("../test_images/krisWu_contrast.jpg", cropped_width, cropped_width, gray_channels, img, 100);

    // mask image to circle, need to use a image that does not have white background
    crop_circle(img, cropped_width, cropped_width/2);
    stbi_write_jpg("../test_images/krisWu_circle.jpg", cropped_width, cropped_width, gray_channels, img, 100);

    // invert image so that darker pixel has greater value
    unsigned char* inverted_img = (unsigned char*)malloc(cropped_size);
    invert_image(img, inverted_img, cropped_size, gray_channels);
    stbi_write_jpg("../test_images/krisWu_inverted.jpg", cropped_width, cropped_width, gray_channels, inverted_img, 100);

    TIMER(prevTime, "preprocessing")
    // find pin coordinates
    int x_coords[numPins];
    int y_coords[numPins];
    find_pinCords(numPins, cropped_width/2, cropped_width, x_coords, y_coords);
    plot_pinCords(img, numPins, cropped_width, x_coords, y_coords);
    stbi_write_jpg("../test_images/krisWu_pins.jpg", cropped_width, cropped_width, gray_channels, img, 100);
    TIMER(prevTime, "finding pins");

    // // test draw lines
    // int line_x[cropped_width];
    // int line_y[cropped_width];
    // int line_length;
    // find_linePixels(x_coords[8], y_coords[8], x_coords[2], y_coords[2], line_x, line_y, &line_length, cropped_width);
    // drawLine(img, line_x, line_y, line_length, cropped_width);
    // stbi_write_jpg("../test_images/krisWu_line_test.jpg", cropped_width, cropped_width, gray_channels, img, 100);

    size_t currNorm = 0;
    int line_length;
    int line_x[cropped_width];
    int line_y[cropped_width];
    std::queue<int> found_p1, found_p2, tmp_found_p1, tmp_found_p2;
    int linesFound = 0;
    bool isAdd = true;
    unsigned char* constructed_img = (unsigned char*)malloc(cropped_size);
    // initialize the constructed img to be blank
    for (size_t i = 0; i<cropped_size; i++) {
        constructed_img[i] = 0;
        currNorm += inverted_img[i]*inverted_img[i];
    }
    size_t bestNorm = currNorm;
    bool noAddition = false;
    bool noRemoval = false;

    while (true) {
        int bestPin1 = 0;
        int bestPin2 = 0;
        if (isAdd) {
            // find the line starting from pin that has the biggest norm reduction
            for (size_t i = 0; i<numPins; i++) {
                for (size_t j = 0; j<i; j++) {
                    // printf("cp1, line_length: %d\n", line_length);
                    find_linePixels(x_coords[i], y_coords[i], x_coords[j], y_coords[j], line_x, line_y, &line_length, cropped_width);
                    // printf("cp2, line_length: %d\n", line_length);
                    size_t tmp_norm = l2_norm(constructed_img, inverted_img, cropped_size, cropped_width, line_x, line_y, line_length, true);
                    // printf("i: %lu,j: %lu, tmp_norm: %lu, bestNorm: %lu\n", i, j, tmp_norm, bestNorm);
                    if (tmp_norm < bestNorm) {
                        noRemoval = false;
                        // printf("tmp_norm < bestNorm\n");
                        bestPin1 = i;
                        bestPin2 = j;
                        bestNorm = tmp_norm;
                    }
                    // printf("cp3\n");
                }
            }
            if (bestPin1 == bestPin2) { // no line can make norm any smaller
                printf("1 pass of adding is done \n");
                // isAdd = false; // try deleting lines
                // noRemoval = true; // remove this
                // noAddition = true;
                break;
                continue;
            } 
            printf("found pin1: %d, pin2: %d\n", bestPin1, bestPin2);
            // whiten the pixels covered by line
            find_linePixels(x_coords[bestPin1], y_coords[bestPin1], x_coords[bestPin2], y_coords[bestPin2], line_x, line_y, &line_length, cropped_width);
            add_line2Img(constructed_img, img, cropped_width, line_x, line_y, line_length);
            found_p1.push(bestPin1);
            found_p2.push(bestPin2);
        }
        else { //isAdd == false
            printf("inside else\n");
            int firstP1 = found_p1.front();
            int firstP2 = found_p2.front();
            // printf("firstP1: %d, firstP2: %d\n", firstP1, firstP2);
            int p1 = firstP1;
            int p2 = firstP2;
            do {
                // printf("p1: %d, p2: %d \n", p1, p2);
                found_p1.pop();
                found_p2.pop();
                find_linePixels(x_coords[p1], y_coords[p1], x_coords[p2], y_coords[p2], line_x, line_y, &line_length, cropped_width);
                size_t tmp_norm = l2_norm(constructed_img, inverted_img, cropped_size, cropped_width, line_x, line_y, line_length, false);
                if (tmp_norm < bestNorm) {
                    noAddition = false;
                    remove_lineFromImg(constructed_img, inverted_img,cropped_width, line_x, line_y, line_length);
                    bestNorm = tmp_norm;
                    if (p1 == firstP1 && p2 == firstP2) {
                        firstP1 = found_p1.front();
                        firstP2 = found_p2.front();
                        printf("updating firstP1: %d, firstP2: %d \n", firstP1, firstP2);
                    }
                }else {
                    found_p1.push(p1);
                    found_p2.push(p2);
                }
                p1 = found_p1.front();
                p2 = found_p2.front();
            } while (p1 != firstP1 || p2 != firstP2);
            isAdd = true;
            noRemoval = true;
            printf("1 pass of removing is done\n");
        }
        // update norm
        if (noAddition && noRemoval) break;
        currNorm = bestNorm;
    }
    stbi_write_jpg("../test_images/krisWu_lines.jpg", cropped_width, cropped_width, gray_channels, img, 100);
    unsigned char* inverted_constructed_img = (unsigned char*)malloc(cropped_size);
    invert_image(constructed_img, inverted_constructed_img, cropped_size, gray_channels);
    stbi_write_jpg("../test_images/krisWu_justlines.jpg", cropped_width, cropped_width, gray_channels, inverted_constructed_img, 100);
    TIMER(prevTime, "finding edges")
}

