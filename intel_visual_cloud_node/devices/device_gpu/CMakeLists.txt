# Copyright (c) 2015, Intel Corporation
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

# Source files
set (CORE_SRC
      "core/layer_arithmetic_opencl.cpp" 
      "core/layer_convolution_opencl.cpp" 
      "core/layer_pooling_opencl.cpp" 
      "core/layer_softmax_opencl.cpp" 
      "core/layer_normalization_opencl.cpp" 
      "core/layer_fully_connected_opencl.cpp" 
      "core/layer_norm_linear_single_opencl.cpp" 
      "core/layer_conv_maxpooling_opencl.cpp" 
      "core/toolkit_opencl.cpp" 
      "core/layers_opencl.h"
      "core/ocl_kernels.cpp")
      
set (API_INTERNAL_SRC
      "api_internal/nn_device_api.cpp"
      "api_internal/nn_device_interface_0_functions.cpp"
      "api_internal/nn_device_interface_0_functions.h"
      "api_internal/nn_device_interface_0_internal.h"
      "../common/nn_device_interface_0_internal_common.cpp"
      "../common/nn_workload_data.cpp"
      "../common/nn_workload_data.h")

set (DEVICE_API
      "../api/nn_device_api.h"
      "../api/nn_device_interface_0.h"
     )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("core" FILES ${CORE_SRC})
source_group("api_internal" FILES ${API_INTERNAL_SRC})
source_group("api" FILES ${DEVICE_API})
# Create .dll/.so in Release/Debug and .lib in DebugULT
if( CMAKE_BUILD_TYPE STREQUAL "Release" )
    add_library(device_gpu MODULE ${CORE_SRC} ${API_INTERNAL_SRC} ${DEVICE_API})
elseif( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    add_library(device_gpu MODULE ${CORE_SRC} ${API_INTERNAL_SRC} ${DEVICE_API})
elseif( CMAKE_BUILD_TYPE STREQUAL "DebugULT" )
    add_library(device_gpu STATIC ${CORE_SRC} ${API_INTERNAL_SRC} ${DEVICE_API})
else()
    message("Unknown configuration")
endif()

# Get the list of directories where libs could be stored
file(GLOB OPENCL_LIB_DIRS $ENV{INTELOCLSDKROOT}/../**/lib64/)

# Order of searching is that first SDK is checked for OpenCL headers/libs then 
# headers from project are taken and libs from hardcoded system locations
find_path(OPENCL_INCLUDE_DIRS NAMES CL/cl.hpp HINTS $ENV{INTELOCLSDKROOT}/include PATHS ../../ )
find_library(OPENCL_LIBRARIES NAMES OpenCL OpenCL.lib HINTS $ENV{INTELOCLSDKROOT}/lib/x64 $ENV{INTELOCLSDKROOT}
 ${OPENCL_LIB_DIRS} PATHS /lib64 /usr/lib )

include_directories(${OPENCL_INCLUDE_DIRS})

target_link_libraries(device_gpu ${OPENCL_LIBRARIES} )
