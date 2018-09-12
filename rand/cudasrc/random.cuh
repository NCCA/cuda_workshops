#ifndef _RANDOM_CUH
#define _RANDOM_CUH

#include <vector>
#include <stdio.h>
#include <time.h>
#include <vector>

/// Fill up a vector on the device with n floats. Memory is arrumed to have been preallocated.
int randFloats(float *&/*devData*/, const size_t /*n*/);

/// Given an stl vector of floats, fill it up with random numbers
int randFloatsToCPU(std::vector<float>&);

#endif //_RANDOM_CUH
