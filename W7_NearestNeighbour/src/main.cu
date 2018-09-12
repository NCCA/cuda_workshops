
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

#include<sys/time.h>

// My own include function to generate some randomness
#include "random.cuh"


/// The number of points to generate within 0,1
#define NUM_POINTS 50

/// The resolution of our grid (dependent on the radius of influence of each point)
#define GRID_RESOLUTION 4

/// The null hash indicates the point isn't in the grid (this shouldn't happen if your extents are correctly chosen)
#define NULL_HASH UINT_MAX

/**
  * Find the cell hash of each point. The hash is returned as the mapping of a point index to a cell.
  * If the point isn't inside any cell, it is set to NULL_HASH. This may have repercussions later in
  * the code.
  * \param Px The array of x values
  * \param Py The array of y values
  * \param Pz the array of z values
  * \param hash The array of hash output
  * \param N The number of points (dimensions of Px,Py,Pz and hash)
  * \param res The resolution of our grid.
  */
__global__ void pointHash(unsigned int *hash,
                          const float *Px,
                          const float *Py,
                          const float *Pz,
                          const unsigned int N,
                          const unsigned int res) {
    // Compute the index of this thread: i.e. the point we are testing
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Note that finding the grid coordinates are much simpler if the grid is over the range [0,1] in
        // each dimension and the points are also in the same space.
        int gridPos[3];
        gridPos[0] = floor(Px[idx] * res);
        gridPos[1] = floor(Py[idx] * res);
        gridPos[2] = floor(Pz[idx] * res);

        // Test to see if all of the points are inside the grid
        bool isInside = true;
        unsigned int i;
        for (i=0; i<3; ++i)
            if ((gridPos[i] < 0) || (gridPos[i] > res)) {
                isInside = false;
            }

        // Write out the hash value if the point is within range [0,1], else write NULL_HASH
        if (isInside) {
            hash[idx] = gridPos[0] * res * res + gridPos[1] * res + gridPos[2];
        } else {
            hash[idx] = NULL_HASH;
        }
        // Uncomment the lines below for debugging. Not recommended for 4mil points!
        //printf("pointHash<<<%d>>>: P=[%f,%f,%f] gridPos=[%d,%d,%d] hash=%d\n",
        //       idx, Px[idx], Py[idx], Pz[idx],
        //       gridPos[0], gridPos[1], gridPos[2], hash[idx]);
    }
}

/**
  * Compute the grid cell occupancy from the input vector of grid hash values. Note that the hash[]
  * vector doesn't need to be presorted, but performance will probably improve if the memory is
  * contiguous.
  * \param cellOcc A vector, size GRID_RES^3, which will contain the occupancy of each cell
  * \param hash A vector, size NUM_POINTS, which contains the hash of the grid cell of this point
  * \param nCells The size of the cellOcc vector (GRID_RES^3)
  * \param nPoints The number of points (size of hash)
  */
__global__ void countCellOccupancy(unsigned int *cellOcc,
                                   unsigned int *hash,
                                   unsigned int nCells,
                                   unsigned int nPoints) {
    // Compute the index of this thread: i.e. the point we are testing
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform a sanity check and then atomic add to the occupancy count of the relevant cell
    if ((idx < nPoints) && (hash[idx] < nCells)) {
        atomicAdd(&(cellOcc[hash[idx]]), 1);
    }
}

/**
 * Host main routine
 */
int main(void) {
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

    // This vector will hold the grid cell occupancy (set to zero)
    thrust::device_vector<unsigned int> d_cellOcc(GRID_RESOLUTION*GRID_RESOLUTION*GRID_RESOLUTION, 0);

    // This vector will hold our hash values, one for each point
    thrust::device_vector<unsigned int> d_hash(NUM_POINTS);
    //thrust::copy(d_hash.begin(), d_hash.end(), std::ostream_iterator<unsigned int>(std::cout, " "));

    // Typecast some raw pointers to the data so we can access them with CUDA functions
    unsigned int * d_hash_ptr = thrust::raw_pointer_cast(&d_hash[0]);
    unsigned int * d_cellOcc_ptr = thrust::raw_pointer_cast(&d_cellOcc[0]);
    float * d_Px_ptr = thrust::raw_pointer_cast(&d_Px[0]);
    float * d_Py_ptr = thrust::raw_pointer_cast(&d_Py[0]);
    float * d_Pz_ptr = thrust::raw_pointer_cast(&d_Pz[0]);

    // The number of threads per blockshould normally be determined from your hardware, but 1024
    // is pretty standard. Remember that each block will be assigned to a single SM, with it's
    // own local memory.
    unsigned int nThreads = 1024;
    unsigned int nBlocks = NUM_POINTS / nThreads + 1;

    struct timeval tim;
    double t1, t2;
    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec/1000000.0);

    // The special CUDA syntax below executes our parallel function with the specified parameters
    // using the number of blocks and threads provided.
    pointHash<<<nBlocks, nThreads>>>(d_hash_ptr, d_Px_ptr, d_Py_ptr, d_Pz_ptr,
                                     NUM_POINTS,
                                     GRID_RESOLUTION);

    // Make sure all threads have wrapped up before completing the timings
    cudaThreadSynchronize();

    // Now we can sort our points to ensure that points in the same grid cells occupy contiguous memory
    thrust::sort_by_key(d_hash.begin(), d_hash.end(),
                        thrust::make_zip_iterator(
                            thrust::make_tuple( d_Px.begin(), d_Py.begin(), d_Pz.begin())));

    // Make sure all threads have wrapped up before completing the timings
    cudaThreadSynchronize();

    // Now we can count the number of points in each grid cell
    countCellOccupancy<<<nBlocks, nThreads>>>(d_cellOcc_ptr, d_hash_ptr, d_cellOcc.size(), d_hash.size());

    // Make sure all threads have wrapped up before completing the timings
    cudaThreadSynchronize();

    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    std::cout << "Grid sorted "<<NUM_POINTS<<" points into grid of "<<GRID_RESOLUTION*GRID_RESOLUTION*GRID_RESOLUTION<<" cells in " << t2-t1 << "s\n";

    // Only dump the debugging information if we have a manageable number of points.
    if (NUM_POINTS <= 100) {
        thrust::copy(d_hash.begin(), d_hash.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
        std::cout << "\n";
        thrust::copy(d_cellOcc.begin(), d_cellOcc.end(), std::ostream_iterator<unsigned int>(std::cout, " "));
    }
    return 0;
}

