#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "src/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/stb_image/stb_image_write.h"

size_t truncate(int32_t value)
{   
    if(value < 0) return 0;
    if(value > 255) return 255;

    return value;
}

int main(void) {
    int width, height, channels;
    unsigned char *img = stbi_load("test_images/sky.jpeg", &width, &height, &channels, 0);
    if(img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

    // Convert the input image to gray
    size_t img_size = width * height * channels;
    int gray_channels = channels == 4 ? 2 : 1;
    size_t gray_img_size = width * height * gray_channels;

    unsigned char *gray_img = (unsigned char*)malloc(gray_img_size);
    if(gray_img == NULL) {
        printf("Unable to allocate memory for the gray image.\n");
        exit(1);
    }

    for(unsigned char *p = img, *pg = gray_img; p != img + img_size; p += channels, pg += gray_channels) {
         *pg = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
         if(channels == 4) {
             *(pg + 1) = *(p + 3);
         }
    }
    stbi_write_jpg("test_images/sky_gray.jpeg", width, height, gray_channels, gray_img, 100);
    
    int contrast = 100;
    float factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));
    for(unsigned char *pg = gray_img; pg != gray_img + gray_img_size; pg += gray_channels) {
        uint8_t p = (uint8_t)*pg;
        int32_t np = (int32_t)(uint32_t)p;
        *pg = (uint8_t)truncate((factor * (np - 128) + 128));
    }
    
    stbi_write_jpg("test_images/sky_gray_contrast.jpeg", width, height, gray_channels, gray_img, 100);
}

