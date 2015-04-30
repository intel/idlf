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
file (GLOB NN_DATA_SRC
      "src/*.c" )
      
file (GLOB NN_DATA_EXAMPLES_SRC
      "examples/*.cpp" 
      "examples/*.c")

file (GLOB NN_DATA_INCLUDE
      "../devices/api/nn_workload_data.h" 
	  "../devices/api/nn_workload_data_cpp_wrapper.h" )


# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("nn_data" FILES ${NN_DATA_SRC})
source_group("nn_data_examples" FILES ${NN_DATA_EXAMPLES_SRC})
source_group("include" FILES ${NN_DATA_INCLUDE})

# Properties->C/C++->General->Additional Include Directories
include_directories ("../devices/api/")

set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# targets
add_library(nn_data STATIC ${NN_DATA_SRC} ${NN_DATA_INCLUDE} )
add_executable(nn_data_examples ${NN_DATA_EXAMPLES_SRC} )

# Make this static lib position independent code (needed if this lib is to be part of share object)
set_property(TARGET nn_data PROPERTY POSITION_INDEPENDENT_CODE ON )

# Set library dependencies (DLlib required for unix)
target_link_libraries(nn_data_examples nn_data)
