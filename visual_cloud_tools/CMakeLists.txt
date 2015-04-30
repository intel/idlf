# Copyright (c) 2014, Intel Corporation
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 2.8)

project (visual_cloud_demo)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)

# To Enable C++11 gcc (Linux compiler) require additional flag to be added
if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
add_definitions(-std=c++0x)
endif()
if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
add_definitions(-march=core-avx2)
add_definitions(-no-fma)
add_definitions(-fp-model fast)
else()
add_definitions(-mavx)
add_definitions(-mavx2)
add_definitions(-mfma)
add_definitions(-ffast-math)
endif()
add_definitions(-march=native)
endif()

#add_definitions(-DDEVICE_GPU)   # temporary hack to choose which device to build against
# Support for clang_complete (needed for easier navigation in Vim/Emacs and others supporting clang_complete)
include(./cmake/copy_clang_complete.cmake )
#TODO: consider adding Visual compiler specific optimizations

# Get the directory partial suffix where nn_data is to be stored eg. UnixMk/Release
string(REPLACE ${CMAKE_SOURCE_DIR} "" CONFIG_DIR ${CMAKE_BINARY_DIR})

if(UNIX)
# On Linux Freeimage provided by host OS is used
find_package(FreeImage) 
else()
# For windows we will used FreeImage binary stored at repo
set(NN_DATA_PATH ${CMAKE_SOURCE_DIR}/../intel_visual_cloud_node/${CONFIG_DIR}/bin/${CMAKE_BUILD_TYPE})
find_library(FreeImage_LIBRARIES FreeImage PATHS ${CMAKE_SOURCE_DIR}/FreeImage64)
set(FreeImage_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/FreeImage64")
endif()

set(COMMON_SRC
    FreeImage_wraps.h
    FreeImage_wraps.cpp
    nn_data_tools.h
    nn_data_tools.cpp
    report_maker.h
    report_maker.cpp
    time_control.h
    time_control.cpp
    linux_api_emulation.h
    )


file (GLOB WORKFLOWS_SRC
    workflow_builder.h
    workflow_builder_*.cpp)
    
file (GLOB PRIMITIVE_WORKLOADS_SRC
    primitives_workload.h
    primitives_workload_*.cpp)

include_directories("${CMAKE_SOURCE_DIR}/../intel_visual_cloud_node/devices/api")

#TODO remove those
include_directories("${CMAKE_SOURCE_DIR}/../intel_visual_cloud_node/devices/device_cpu/api_internal")
#END TODO

include_directories(${FreeImage_INCLUDE_DIRS})

add_executable(visual_cloud_demo main.cpp ${COMMON_SRC} ${WORKFLOWS_SRC} )
add_executable(demo_primitives primitives_main.cpp ${COMMON_SRC} ${PRIMITIVE_WORKLOADS_SRC})

target_link_libraries(visual_cloud_demo ${FreeImage_LIBRARIES})
if(UNIX)
target_link_libraries(visual_cloud_demo dl)
endif()

target_link_libraries(demo_primitives ${FreeImage_LIBRARIES})
if(UNIX)
target_link_libraries(demo_primitives dl)
endif()

install(TARGETS visual_cloud_demo DESTINATION ${CMAKE_SOURCE_DIR}/demo_bin PERMISSIONS OWNER_WRITE  OWNER_READ OWNER_EXECUTE WORLD_READ WORLD_EXECUTE)
