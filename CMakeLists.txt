cmake_minimum_required(VERSION 3.18)
project(cuda-password-cracker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)   # RTX 50 series (Blackwell); override with -DCMAKE_CUDA_ARCHITECTURES=90 etc.
endif()

add_executable(sha256 src/brute_sha256.cu)
target_compile_options(sha256 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(zipcrack src/zipcrack.cu)
target_compile_options(zipcrack PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(zipcrack_cpu src/zipcrack_cpu.cpp)
target_compile_options(zipcrack_cpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
