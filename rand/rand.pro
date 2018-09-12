######################################################################
# Automatically generated by qmake (2.01a) Thu Nov 2 10:05:22 2017
######################################################################

TEMPLATE = app
OBJECTS_DIR=obj
TARGET = rand
CONFIG = debug console opengl

# Directories
SOURCES += src/main.cpp
INCLUDEPATH += cudasrc ${CUDA_PATH}/include ${CUDA_PATH}/samples/common/inc ${CUDA_PATH}/include/cuda

LIBS += -L${CUDA_PATH}/lib64 -L/usr/lib/x86_64-linux-gnu -L/usr/lib64/nvidia -lcudadevrt -lcuda -lcudart -lcurand
 
## CUDA_COMPUTE_ARCH - This will enable nvcc to compiler appropriate architecture specific code for different compute versions.
## Multiple architectures can be requested by using a space to seperate. example:
## CUDA_COMPUTE_ARCH = 10 20 30 35
CUDA_COMPUTE_ARCH = 52
 
## CUDA_DEFINES - The seperate defines needed for the cuda device and host methods
CUDA_DEFINES +=
 
## CUDA_DIR - the directory of cuda such that CUDA\<version-number\ contains the bin, lib, src and include folders
CUDA_DIR= ${CUDA_PATH}
 
## CUDA_SOURCES - the source (generally .cu) files for nvcc. No spaces in path names
CUDA_SOURCES += cudasrc/random.cu
 
## CUDA_LIBS - the libraries to link
CUDA_LIBS = -L${CUDA_PATH}/lib64 -L/usr/lib/x86_64-linux-gnu -lcudart -lcurand -lcuda
 
## CUDA_INC - all incldues needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC+= $$join(INCLUDEPATH,' -I','-I',' ')
 
## NVCC_OPTIONS - any further options for the compiler
NVCC_OPTIONS += --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
 
# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = -ccbin $$(HOST_COMPILER) --compiler-options -fno-strict-aliasing -use_fast_math #--ptxas-options=-v
 
#prepare intermediat cuda compiler
cudaIntr.input = CUDA_SOURCES
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
 
## Tweak arch according to your hw's compute capability
cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G $$CUDA_INC $$NVCCFLAGS -gencode arch=compute_$$CUDA_COMPUTE_ARCH,code=sm_$$CUDA_COMPUTE_ARCH -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 
#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
 
QMAKE_EXTRA_COMPILERS += cudaIntr
 
 
# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o
 
# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G $$CUDA_INC $$NVCCFLAGS -gencode arch=compute_$$CUDA_COMPUTE_ARCH,code=sm_$$CUDA_COMPUTE_ARCH -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
