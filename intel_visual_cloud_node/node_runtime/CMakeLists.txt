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

#Files
file (GLOB RESOURCE_MANAGER_SRC
      "resource_manager/*.h" 
      "resource_manager/*.cpp")

file (GLOB SCHEDULER_SRC
      "scheduler/*.h" 
      "scheduler/*.cpp")
      
file (GLOB COMMON_SRC
      "../common/*.h" 
      "../common/*.cpp")

file (GLOB DEVICE_API
      "../devices/api/*.h")
     
# Main source	 
set  (MAIN_SRC
      "main.cpp")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("" FILES ${MAIN_SRC})
source_group("scheduler" FILES ${SCHEDULER_SRC})
source_group("resource_manager" FILES ${RESOURCE_MANAGER_SRC})
source_group("devices_api" FILES ${DEVICE_API})
source_group("common" FILES ${COMMON_SRC})

# Properties->C/C++->General->Additional Include Directories
include_directories (.)

# Create .exe in Release/Debug and .lib in DebugULT
if( CMAKE_BUILD_TYPE STREQUAL "Release")
    add_executable(node_runtime ${MAIN_SRC} ${SCHEDULER_SRC} ${RESOURCE_MANAGER_SRC} ${COMMON_SRC} ${DEVICE_API})
elseif( CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_executable(node_runtime ${MAIN_SRC} ${SCHEDULER_SRC} ${RESOURCE_MANAGER_SRC} ${COMMON_SRC} ${DEVICE_API})
elseif( CMAKE_BUILD_TYPE STREQUAL "DebugULT")
    add_library(node_runtime STATIC ${MAIN_SRC} ${SCHEDULER_SRC} ${RESOURCE_MANAGER_SRC} ${COMMON_SRC} ${DEVICE_API})
else()
    message("Unknown configuration")
endif()

# Set library dependencies (DLlib required for unix)
target_link_libraries(node_runtime ${CMAKE_DL_LIBS})