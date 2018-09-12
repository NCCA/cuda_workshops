// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// Needed for output functions within the kernel
#include <stdio.h>

// Needed for streaming with thrust iterators 
#include <iostream>

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Include our teapot data
#include <teapot.h>

// Needed to get matrix functions working in CUDA
#include <eigen-nvcc/Dense>

// Note that the resolution has to be at least 4 because we are writing the values into shared
// memory in parallel: if the RES is less than 4 not all the values will be written to the shared
// matrix structure.
// NOTE: My system fails to handle a RES higher than 30 - it _should_ theoretically handle 32. This
//       could be a hardware fault. The result of RES being too high is that your output will just 
//       be full of zeros.
#define RES 30

/** 
 * Compute interpolate points across a series of bezier patches, according to the formula of:
 * http://www.idav.ucdavis.edu/education/CAGDNotes/Matrix-Cubic-Bezier-Patch/Matrix-Cubic-Bezier-Patch.html
 * Note that this uses Eigen for clarity as this neatly wraps up 4x4 matrix multiplications. However,
 * this process means that we get loads of compilation warnings, as the library is very much in beta.
 *
 * This kernel function requires a 2D block of threads, with the dimension in each of the x and y
 * directions indicates the u and v parameters for the interpolation (as you'd expect). However, this
 * implementation has limitations as the hardware has limitations on the block sizes that are manageable.
 * However, this approach is far neater as each block works on a single patch, and we can therefore exploit
 * shared memory when constructing the matrices P_x, P_y and P_z.
 *
 * \param output_vertices The returned vertices. Size of this vector is num_patches*res*res*3, where res is the
 *        resolution of the patch in u or v direction.
 * \param patches The indices of the vertices in each patch. These are grouped in sections of 16 vertices which
 *        are the bezier control points.
 * \param vertices The input vertices, size num_vertices*3. Note we assume that there are enough vertices for the
 *        the patches to index.
 * \param num_patches The number of patches. patches/num_patches should equal 16.
 */
__global__ void tesselate(float *output_vertices,
			    const unsigned int *patches, 
			    const float *vertices,
			    const unsigned int num_patches) {
    // Here is where YOU make it happen
}

/**
 * Host main routine. In this function we'll basically stash the teapot data onto the device using
 * thrust iterators, call the kernel, and then output the result to the screen, formatted for an
 * obj file.
 */
int main(void) {
	// Create our vectors for the device to hold our point data by using
	// array iterators (like STL)
	thrust::device_vector<float> d_vertices(teapot::vertices, teapot::vertices + 3*teapot::num_vertices);

	// Create an array for our patch indices
	thrust::device_vector<unsigned int> d_patches(teapot::patches, teapot::patches + 16*teapot::num_patches);

	// Create output data structure (there will be (x,y,z) values in both the u and v directions along the patch)
	thrust::device_vector<float> d_output_vertices(RES*RES*3*teapot::num_patches);

	// Define the dimension of the block - the u and v coordinates of the sampled point are given
	// by the threadIdx.x/y.
	dim3 blockDim(RES, RES, 1);

	// Call our kernel function to sample points on the bezier patch
	tesselate<<<teapot::num_patches, blockDim>>>(thrust::raw_pointer_cast(&d_output_vertices[0]),
	  					thrust::raw_pointer_cast(&d_patches[0]),
						thrust::raw_pointer_cast(&d_vertices[0]),
						teapot::num_patches);

	// Dump the data in obj format to cout (use the pipe ">" command to make an obj file)
	thrust::device_vector<float>::iterator dit = d_output_vertices.begin();
	unsigned int i;
	for (i=0; i<RES*RES*teapot::num_patches; ++i) {
		std::cout << "v ";
		thrust::copy(dit, dit+3, std::ostream_iterator<float>(std::cout, " "));
		std::cout << "\n";
		dit += 3;
	}

	// Exit politely
	return 0;
}

