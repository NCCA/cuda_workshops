
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data, accessible to the host
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
float reduceCPU(float *data, int size)
{
    float sum = data[0];
    float c = (float)0.0;

    for (int i = 1; i < size; i++)
    {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

void fillCPU(float *data, unsigned int size) {
    float inv_randmax = rand() / float(RAND_MAX);
    unsigned int i;
    for (i=0; i<size; ++i) data[i] = rand()*inv_randmax;
}


/**
 * Host main routine
 */
int main(int argc, char **argv) {
    // Process command line
    unsigned int size = 100;
    if (argc > 1) {
        size = atoi(argv[1]);
    }

    // Some stuff we need to perform timings
    struct timeval tim;
    double before, after;

    // Somewhere to store the result
    float resultCPU;

    // Create some data
    float *h_rand = (float*) malloc(sizeof(float) * size);

    fillCPU(h_rand, size);

    // Copy the random vector onto the host so we can compute the result on the CPU
    gettimeofday(&tim, NULL);
    before=tim.tv_sec+(tim.tv_usec * 1e-6);

    resultCPU = reduceCPU(h_rand, size);
    gettimeofday(&tim, NULL);
    after=tim.tv_sec+(tim.tv_usec * 1e-6);
    double reduceCPU_time = after-before;
    std::cout << "Reduced vector of length "<<size<<" on CPU in "<< reduceCPU_time<< "s, resulting in "<<resultCPU << "\n";

    return EXIT_SUCCESS;
}

