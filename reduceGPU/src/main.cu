
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <device_functions.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>

// Needed for output functions within the kernel
#include <stdio.h>

// printf() is only supported
// for devices of compute capability 2.0 and higher

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// Min macro may be defined externally
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif


// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <sys/time.h>

// My own include function to generate some randomness
#include "random.cuh"


/**
 * \brief nextPow2 Utility function to determine the number of threads in the block
 * \param x Input value
 * \return The next power of 2 after x
 * \see getNumBlocksAndThreads
 */
unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/**
 * \brief Compute the number of threads and blocks to use for the given reduction kernel. We set to the minimum of maxThreads and n.
 * \param n The size of the vector / problem
 * \param blocks number of blocks to use (returned)
 * \param threads number of threads to use (returned)
 */
template<class T>
void getNumBlocksAndThreads(int n, int &blocks, int &threads, int &smem) {

    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    threads = (n < prop.maxThreadsPerBlock) ? nextPow2(n) : prop.maxThreadsPerBlock;
    blocks = (n + threads - 1) / threads;

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    smem = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    // Should probably do some error checking here
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data, accessible to the host
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;

    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int n) {
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * Perform the reduction on the GPU, given a device friendly pointer. This will run the reduction log_m(n) times, where m is
 * the number of threads in the block.
 * \param data Pointer to data, accessible to host
 * \param size size of the data
 * \returns The sum of the data
 */
template <class T>
T reduceGPU(T *data, unsigned int size) {
    T *d_idata = data; // A pointer to the input data
    T *d_odata = NULL; // A pointer to our output data, which will swap after each iteration
    T *d_swap = NULL; // A pointer used for swapping

    unsigned int n = size; // The size will reduce after each iteration
    int nBlocks, nThreads; // The number of blocks and threads changes with each iteration
    int sMemSize;          // This is the amount of shared memory to be allocated to a block

    // Keep tabs on whether the pointer has been flipped between d_idata and d_odata so we don't delete the
    // wrong pointer when we cleanup
    bool swapped = false;



    // Our reduction loop continues until the problem size become 1, in which case we have found the solution
    while (n > 1) {
        // Determine the number of blocks and threads to use within this run
        getNumBlocksAndThreads<T>(n, nBlocks, nThreads, sMemSize);
//        std::cerr << "getNumBlocksAndThreads("<<n<<","<<nBlocks<<","<<nThreads<<")\n";

        // We only allocate the memory once - no point in wasting time here
        if (d_odata == NULL) checkCudaErrors(cudaMalloc((void **) &d_odata, nBlocks*sizeof(T)));

        // Perform a single reduction step
        reduce0<T><<<nBlocks,nThreads,sMemSize>>>(d_idata, d_odata, n);

        // Make sure all threads have wrapped up before proceeding to the next iteration
        cudaThreadSynchronize();

        // Now we've reduced, the new value of n can be deduced from the number of blocks
        n = nBlocks;

        // We can swap the pointers
        d_swap = d_idata; d_idata = d_odata; d_odata = d_swap; d_swap = NULL;
        swapped = !swapped;
    }
    // After our last iteration, we can be satisfied that d_idata[0] contains our total
    T ret_val;
    checkCudaErrors(cudaMemcpy(&ret_val, d_idata, sizeof(T), cudaMemcpyDeviceToHost));
    if (swapped) {
        if (d_idata != NULL) checkCudaErrors(cudaFree(d_idata));
    } else {
        if (d_odata != NULL) checkCudaErrors(cudaFree(d_odata));
    }


    return ret_val;
}


/**
 * Host main routine
 */
int main(int argc, char **argv) {
    // Process command line
    unsigned int VECTOR_SIZE = 100;
    if (argc > 1) {
        VECTOR_SIZE = atoi(argv[1]);
    }

    // Some stuff we need to perform timings
    struct timeval tim;
    double before, after;

    // Somewhere to store the result
    float resultGPU, resultCPU;

    // First thing is we'll generate a big old vector of random numbers for the purposes of
    // fleshing out our point data. This is much faster to do in one step than 2 seperate
    // steps.
    float *d_Rand, *h_Rand;
    checkCudaErrors(cudaMalloc((void**) &d_Rand, VECTOR_SIZE*sizeof(float))); // We need to deep copy the input data lest we mangle it.
    randFloats(d_Rand, VECTOR_SIZE);

    // Copy the random vector onto the host so we can compute the result on the CPU
    h_Rand = (float*) malloc(sizeof(float)*VECTOR_SIZE);
    checkCudaErrors(cudaMemcpy(h_Rand, d_Rand, sizeof(float)*VECTOR_SIZE, cudaMemcpyDeviceToHost));
    gettimeofday(&tim, NULL);
    before=tim.tv_sec+(tim.tv_usec * 1e-6);
    resultCPU = reduceCPU<float>(h_Rand, VECTOR_SIZE);
    gettimeofday(&tim, NULL);
    after=tim.tv_sec+(tim.tv_usec * 1e-6);
    double reduceCPU_time = after-before;
    std::cout << "Reduced vector of length "<<VECTOR_SIZE<<" on CPU in "<< reduceCPU_time<< "s, resulting in "<<resultCPU << "\n";

    //  Now compute result on the GPU
    gettimeofday(&tim, NULL);
    before=tim.tv_sec+(tim.tv_usec * 1e-6);
    resultGPU = reduceGPU<float>(d_Rand, VECTOR_SIZE);
    gettimeofday(&tim, NULL);
    after=tim.tv_sec+(tim.tv_usec * 1e-6);
    double reduceGPU_time = after-before;

    std::cout << "Reduced vector of length "<<VECTOR_SIZE<<" on GPU in "<<reduceGPU_time<< "s, resulting in "<<resultGPU << ": a "<<reduceCPU_time/reduceGPU_time<<"X speedup\n";
    checkCudaErrors(cudaFree(d_Rand));
    return EXIT_SUCCESS;
}

