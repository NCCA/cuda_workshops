
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
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


/**
 * Host main routine
 */
int main(void) {
    // Some stuff we need to perform timings
    struct timeval tim;
    double before, after;

    // Time a function and output the result
    gettimeofday(&tim, NULL);
    before=tim.tv_sec+(tim.tv_usec * 1e-6);
    sleep(3); // Your function here
    gettimeofday(&tim, NULL);
    after=tim.tv_sec+(tim.tv_usec * 1e-6);
    std::cout << "Something took "<<after-before<< "s\n";

    // Close up shop
    return EXIT_SUCCESS;
}

