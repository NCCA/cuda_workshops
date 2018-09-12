#include "random.cuh"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <iterator>

int main() {
    std::vector<float> rands(100);
    randFloatsToCPU(rands);
    std::copy(rands.begin(), rands.end(), std::ostream_iterator<float>(std::cout, " "));
    return EXIT_SUCCESS;
}
