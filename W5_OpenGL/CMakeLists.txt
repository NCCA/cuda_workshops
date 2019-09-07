cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
add_executable(opengl  src/main.cu src/trackball.cpp ../shared/src/shader.cpp)
project(W5_OpenGL LANGUAGES CXX CUDA)
include_directories(include ../shared/include/ /usr/local/cuda-9.0/samples/common/inc/)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL )
if (OPENGL_FOUND)
    include_directories(${OPENGL_INCLUDE_DIRS})
    link_libraries(${OPENGL_LIBRARIES})
endif()

target_link_libraries(opengl  curand GLEW glut ${OPENGL_LIBRARIES})
