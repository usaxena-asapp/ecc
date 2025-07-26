cmake_minimum_required(VERSION 3.14)

project(GPU_ECC LANGUAGES CUDA CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags for Release and Debug builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# specify boost configuration for Linux
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME OFF)  # Changed from ON to OFF for Linux
set(Boost_USE_MULTITHREADED ON)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set CUDA compiler flags for Linux/GCC
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler=-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -Xcompiler=-g")

# Find required packages
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_package(OpenMP REQUIRED)
find_package(OpenCV COMPONENTS core REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Link directories
link_directories(${Boost_LIBRARY_DIRS})

# Create executable
add_executable(GPU_ECC main.cpp routines.cpp utility.cpp kernel.cu)

# Set CUDA properties for the target
set_target_properties(GPU_ECC PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "native"  # Auto-detect GPU architecture
)

# Link libraries
target_link_libraries(GPU_ECC PUBLIC 
    OpenMP::OpenMP_CXX 
    ${Boost_LIBRARIES} 
    ${OpenCV_LIBS}
    ${CUDART_LIBRARY}
)
