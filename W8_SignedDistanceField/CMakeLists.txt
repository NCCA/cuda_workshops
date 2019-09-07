cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(W8_SignedDistanceField LANGUAGES CXX CUDA)
include_directories(../shared/include  /usr/local/cuda-9.0/samples/common/inc/) 
add_executable(sdf src/main.cu)
target_link_libraries(sdf png) 

