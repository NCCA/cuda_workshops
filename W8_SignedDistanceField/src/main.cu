
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "helper_math.h"
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <device_functions.h>

// Needed for output functions within the kernel
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Needed for libpng example
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#define PNG_DEBUG 3
#include <png.h>

// A prototype for the PNG image writer function, which I'll hide at the end
int writeImage(const char *filename,
               const int width,
               const int height,
               const float *buffer,
               const char *title);
               
/**
 * Return the signed distance to the boundary. Plenty of other fun awesomeness to be found here:
 * https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
 */
 __device__ float signedDistance(const float3 &pos) {
    // The default SDF is centered at the origin, so we need to transform the point as our cube centre is at [0.5,0.5,0.5]. The top corner
    // is also given by [.5,.5,.5]. The final result is the distance of the vector from pos transformed to the top right quadrant to the corner
    // of the cube. Note that the distance is SIGNED - negative means inside the cube, positive means outside.
    float3 b = make_float3(0.5f,0.5f, 0.5f);
    
    float3 zero = make_float3(0.0f, 0.0f, 0.0f);
    float3 d = fabs(pos - b) - b;
    float d_box = fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0) + length(fmaxf(d,zero));
    
    // The result below is for a sphere bounding the fluid
    float r = 0.5f; // The radius of the sphere
    float d_sphere = length(pos - make_float3(0.5f)) - r;    
    return d_sphere;
}

/**
 * Compute the normal at a position based on the SDF function.
 */
__device__ float3 signedDistanceNormal(const float3 &pos) {
    float eps = 0.0001f;
            
    // Assume that the normal evaluated at this point in the SDF is close enough to the surface normal (a mistake?)
    return normalize(make_float3(signedDistance(make_float3(pos.x + eps, pos.y, pos.z)) - signedDistance(make_float3(pos.x - eps, pos.y, pos.z)),
                                 signedDistance(make_float3(pos.x, pos.y + eps, pos.z)) - signedDistance(make_float3(pos.x, pos.y - eps, pos.z)),
                                 signedDistance(make_float3(pos.x, pos.y, pos.z + eps)) - signedDistance(make_float3(pos.x, pos.y, pos.z - eps))));
    
}

/**
 * Taken from here https://www.iquilezles.org/www/articles/palettes/palettes.htm
 */
__device__ float3 palette(const float t,  const float3 a, const float3 b, const float3 c, const float3 d )
{
    float3 tmp = c*t+d; // This is needed for cos isn't defined on float3's
    return a + b*make_float3(cos(6.28318*tmp.x),cos(6.28318*tmp.y),cos(6.28318*tmp.z));
}

/**
 * Our global SDF colour field generator. Note that this is  
 */
__global__ void colourSDF(float *buffer) {
    // The first thing we need to do is determine the pixel in SDF coordinates, i.e. between 0 and 1    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float3 pixelPos = make_float3(float(x)/1023.0, 
                                  float(y)/1023.0,
                                  0.5f);

    // Write the value calculated from our signed distance to the buffer
    float3 colour =      palette(signedDistance(pixelPos) * 10.0f,
                                 make_float3(0.5, 0.5, 0.5),	
                                 make_float3(0.5, 0.5, 0.5), 
                                 make_float3(1.0, 1.0, 1.0), 
                                 make_float3(0.00, 0.10, 0.20));
    buffer[(x + y*1024)*3 + 0] = colour.x;
    buffer[(x + y*1024)*3 + 1] = colour.y;
    buffer[(x + y*1024)*3 + 2] = colour.z;
}

/**
 * Host main routine
 */
int main(void) {
    // Define the size of the blocks and grid to create a 1024x1024 image, divided into blocks of 8x8 (64 threads)
    dim3 blockDim(8,8,1);
    dim3 gridDim(128,128,1);

    float *d_buffer;

    // This is going to hold our 1024^2 image, and also all three colour channels (R,G & B)
    uint bufferSize = sizeof(float)*1024*1024*3;
    if (cudaMalloc(&d_buffer, bufferSize) != cudaSuccess) {
        // Throw our toys here
        printf ("Error: cannot allocate CUDA memory\n");
        return EXIT_FAILURE;
    }

    // If it's all good, use our kernel to render to the image
    colourSDF<<<gridDim, blockDim>>>(d_buffer);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: CUDA kernel launch failed. Reason: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    cudaThreadSynchronize();

    // Now that the data is theoretically written to, we can copy it back
    float *h_buffer = (float*) malloc(bufferSize);
    if (h_buffer == NULL || cudaMemcpy(h_buffer, d_buffer, bufferSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
        // Throw toys!
        printf ("Error: cannot allocate copy memory from CUDA\n");
        return EXIT_FAILURE;
    }

    // Now create our output image
    if (writeImage("image.png", 1024, 1024, h_buffer, "Signed Distance Fields") != 0) {
        // Throw toys!
        printf ("Error: write image file\n");
        return EXIT_FAILURE;
    }

    // If we got this far, everything went well. Close up shop.
    cudaFree(d_buffer);
    free(h_buffer);

    // Close up shop
    return EXIT_SUCCESS;
}

/** 
 * Write out an image with libPNG. Taken from http://www.labbookpages.co.uk/software/imgProc/libPNG.html
 */

// This bastardised version was necessary to keep compatibility with the dodgy example
inline void setRGB(png_byte *ptr, const float *colour)
{
    ptr[0] = (int) 256.0f * colour[0];
    ptr[1] = (int) 256.0f * colour[1];
    ptr[2] = (int) 256.0f * colour[2];	
}

// This function was lifted almost entirely from the source above. It is old school C: don't use goto's in your code!
int writeImage(const char *filename,
               const int width,
               const int height,
               const float *buffer,
               const char *title)
{
    int code = 0;
    FILE *fp = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep row = NULL;
    // Open file for writing (binary mode)
    fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "Could not open file %s for writing\n", filename);
        code = 1;
        goto finalise;
    }
    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL)
    {
        fprintf(stderr, "Could not allocate write struct\n");
        code = 1;
        goto finalise;
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        fprintf(stderr, "Could not allocate info struct\n");
        code = 1;
        goto finalise;
    }
    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        fprintf(stderr, "Error during png creation\n");
        code = 1;
        goto finalise;
    }
    png_init_io(png_ptr, fp);

    // Write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Set title
    if (title != NULL)
    {
        png_text title_text;
        title_text.compression = PNG_TEXT_COMPRESSION_NONE;
        title_text.key = (char*) "Title";
        title_text.text = (char*) title;
        png_set_text(png_ptr, info_ptr, &title_text, 1);
    }

    png_write_info(png_ptr, info_ptr);
    // Allocate memory for one row (3 bytes per pixel - RGB)
    row = (png_bytep)malloc(3 * width * sizeof(png_byte));

    // Write image data
    int x, y;
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            setRGB(&(row[x * 3]), &buffer[3*(y * width + x)]);
        }
        png_write_row(png_ptr, row);
    }

    // End write
    png_write_end(png_ptr, NULL);
finalise:
    if (fp != NULL)
        fclose(fp);
    if (info_ptr != NULL)
        png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    if (png_ptr != NULL)
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    if (row != NULL)
        free(row);

    return code;
}