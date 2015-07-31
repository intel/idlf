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

# Maps to Visual Studio solution file (intel_visual_cloud_node.sln)
# The solution will have all targets (exe, lib, dll) 
# as Visual Studio projects (.vcproj)


# Remove modules lib prefix for Unix
set( CMAKE_SHARED_MODULE_PREFIX "")


# Define to distinguish among release and debug builds
if( NOT (CMAKE_BUILD_TYPE STREQUAL "Release") )
    add_definitions("-DDEBUG")
endif( NOT (CMAKE_BUILD_TYPE STREQUAL "Release") )

if( CMAKE_BUILD_TYPE STREQUAL "Release")
    if(UNIX)
        CHECK_CXX_COMPILER_FLAG("-march=core-avx2" COMPILER_SUPPORTS_ICC_AVX2)
        if(COMPILER_SUPPORTS_ICC_AVX2)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2")
        endif()
        CHECK_CXX_COMPILER_FLAG("-xCORE-AVX2" COMPILER_SUPPORTS_xCORE_AVX2)
        if(COMPILER_SUPPORTS_xCORE_AVX2)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCORE-AVX2")
        endif()
        CHECK_CXX_COMPILER_FLAG("-no-fma" COMPILER_SUPPORTS_ICC_NO_FMA)
        if(COMPILER_SUPPORTS_ICC_NO_FMA)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-fma")
        endif()
        CHECK_CXX_COMPILER_FLAG("-fp-model fast" COMPILER_SUPPORTS_ICC_FP_MODEL_FAST)
        if(COMPILER_SUPPORTS_ICC_FP_MODEL_FAST)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model fast")
        endif()
        # Intel compiler O3 is increasing build time and not adding much performance value
        # So remove that one from our Release build
        if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
            string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        endif()
    else()
        # windows
        CHECK_CXX_COMPILER_FLAG("/Ot" COMPILER_SUPPORTS_FAVORSPEED)
        CHECK_CXX_COMPILER_FLAG("/Oi" COMPILER_SUPPORTS_INTRINSIC)
        CHECK_CXX_COMPILER_FLAG("/GS-" COMPILER_SUPPORTS_NOSECCHECK)
        CHECK_CXX_COMPILER_FLAG("/QxCORE-AVX2" COMPILER_SUPPORTS_xCORE_AVX2)
        if(COMPILER_SUPPORTS_xCORE_AVX2)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /QxCORE-AVX2")
        endif()
        if(COMPILER_SUPPORTS_FAVORSPEED)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ot")
        endif()
        if(COMPILER_SUPPORTS_INTRINSIC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi")
        endif()
        if(COMPILER_SUPPORTS_NOSECCHECK)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GS-")
        endif()
    endif(UNIX)
endif()

# Set compiler flags and options. 
# For now it is default.
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Commands to output information to the console
# Useful for displaying errors, warnings, and debugging
message ("cxx Flags: " ${CMAKE_CXX_FLAGS})

# Sub-directories where more CMakeLists.txt exist (more vcprojs)
add_subdirectory(cpu)
add_subdirectory(gpu)
