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

# Source files
file (GLOB CORE_SRC
      "core/*.cpp" 
      "core/*.h")

file (GLOB CORE_FIXEDPOINT_SRC
      "core/fixedpoint/*.cpp"
      "core/fixedpoint/*.h")

file (GLOB API_INTERNAL_SRC
      "api_internal/*.cpp" 
      "api_internal/*.h"
      "../common/nn_device_internal.h"
      "../common/nn_device_interface_0_internal_common.cpp"
      "../common/nn_workload_data.cpp"
      "../common/nn_workload_data.h")

file (GLOB DEVICE_API
      "../api/*.h")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("core" FILES ${CORE_SRC})
source_group("core\\fixedpoint" FILES ${CORE_FIXEDPOINT_SRC})
source_group("api_internal" FILES ${API_INTERNAL_SRC})
source_group("api" FILES ${DEVICE_API})

# Create .dll/.so in Release/Debug and .lib in DebugULT
if( CMAKE_BUILD_TYPE STREQUAL "Release" )
    add_library(device_cpu MODULE ${CORE_SRC} ${CORE_FIXEDPOINT_SRC} ${API_INTERNAL_SRC} ${DEVICE_API})
elseif( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    add_library(device_cpu MODULE ${CORE_SRC} ${CORE_FIXEDPOINT_SRC} ${API_INTERNAL_SRC} ${DEVICE_API})
elseif( CMAKE_BUILD_TYPE STREQUAL "DebugULT" )
    add_library(device_cpu STATIC ${CORE_SRC} ${CORE_FIXEDPOINT_SRC} ${API_INTERNAL_SRC} ${DEVICE_API})
else()
    message("Unknown configuration")
endif()

target_link_libraries(device_cpu)