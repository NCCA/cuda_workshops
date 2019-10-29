cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(reduceGPU LANGUAGES CXX CUDA)
include_directories(../shared/include  /usr/local/cuda-9.0/samples/common/inc/) 
add_executable(bankconflicts src/main.cu ../shared/src/random.cu)
target_link_libraries(bankconflicts curand)
 

