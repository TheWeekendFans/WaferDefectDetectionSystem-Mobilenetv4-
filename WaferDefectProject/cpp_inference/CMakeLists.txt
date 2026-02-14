cmake_minimum_required(VERSION 3.18)
project(WaferInference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root directory")
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
    ${TENSORRT_ROOT}/include
    /usr/include
    /usr/include/x86_64-linux-gnu
)

find_library(TENSORRT_LIB nvinfer
    PATHS
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT or install TensorRT dev package.")
endif()

add_library(preprocess_lib STATIC preprocess.cu)
target_include_directories(preprocess_lib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(preprocess_lib PRIVATE CUDA::cudart)

add_executable(wafer_infer main.cpp)
target_include_directories(wafer_infer PRIVATE ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(wafer_infer PRIVATE preprocess_lib ${OpenCV_LIBS} ${TENSORRT_LIB} CUDA::cudart)
