
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

// Needed for output functions within the kernel
#include <stdio.h>

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// Used to compute performance measures
#include<sys/time.h>

// My own include function to generate some randomness
#include "random.cuh"


/// The number of points to generate within 0,1
#define NUM_POINTS 100

/// The resolution of our grid (dependent on the radius of influence of each point)
#define GRID_RESOLUTION 2

/// The null hash indicates the point isn't in the grid (this shouldn't happen if your extents are correctly chosen)
#define NULL_HASH UINT_MAX

/**
 * In this workshop you will need to do part of an important parallel problem in graphics, namely to do a grid sort
 * of a bunch of points. The number of points is given by NUM_POINTS, and code is provided to create three random device
 * arrays d_Px, d_Py, d_Pz containing x,y,z coordinates in a unit cube (range [0,1]).
 * The GRID_RESOLUTION is in each dimension, so for example if resolution is 2, there are 8 cells in total (2^3). You will
 * need to determine the unique pointHash for each point, indicating the integer value of the cell in which the point
 * resides.
 * So for example, if GRID_RESOLUTION is 2, the point (0.2, 0.2, 0.2) lives in cell 0, while (0.7, 0.7, 0.7) lives in
 * cell 7.
 * Compute this on both the CPU and the GPU. Compare these results: what can you say about the performance of the CPU
 * implementation for smaller numbers of points?
 *
 * Richard Southern, 08/01/2013
 **/


/**
 * Host main routine
 */
int main(void) {
    // Needed for timings
    struct timeval tim;
    double t1, t2;

    // First thing is we'll generate a big old vector of random numbers for the purposes of
    // fleshing out our point data. This is much faster to do in one step than 3 seperate
    // steps.
    thrust::device_vector<float> d_Rand(NUM_POINTS*3);
    float * d_Rand_ptr = thrust::raw_pointer_cast(&d_Rand[0]);
    randFloats(d_Rand_ptr, NUM_POINTS*3);

    // We'll store the components of the 3d vectors in separate arrays.
    // This 'structure of arrays' (SoA) approach is usually more efficient than the
    // 'array of structures' (AoS) approach.  The primary reason is that structures,
    // like Float3, don't always obey the memory coalescing rules, so they are not
    // efficiently transferred to and from memory.  Another reason to prefer SoA to
    // AoS is that we don't aways want to process all members of the structure.  For
    // example, if we only need to look at first element of the structure then it
    // is wasteful to load the entire structure from memory.  With the SoA approach,
    // we can chose which elements of the structure we wish to read.
    thrust::device_vector<float> d_Px(d_Rand.begin(), d_Rand.begin()+NUM_POINTS);
    thrust::device_vector<float> d_Py(d_Rand.begin()+NUM_POINTS, d_Rand.begin()+2*NUM_POINTS);
    thrust::device_vector<float> d_Pz(d_Rand.begin()+2*NUM_POINTS, d_Rand.end());

    // Here you will need to create a new array to store your hash result, and then compute the
    // pointHash on both the CPU and GPU. Performance measure routines are provided below.

    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec * 0.0000001);
    // Do something useful here
    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec * 0.0000001);
    std::cout << "Something took " << t2-t1 << "s\n";

    return EXIT_SUCCESS;
}

