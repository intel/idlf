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

# comment (demo) (revision 2)
cmake_minimum_required (VERSION 2.8)

# Maps to Visual Studio solution file (intel_visual_cloud_node.sln)
# The solution will have all targets (exe, lib, dll) 
# as Visual Studio projects (.vcproj)
project (intel_visual_cloud_node)

# Output all files to /bin
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Remove modules lib prefix for Unix
set( CMAKE_SHARED_MODULE_PREFIX "")

# Setup of DebugULT configuration(flags same as Debug)
set( CMAKE_CXX_FLAGS_DEBUGULT "${CMAKE_CXX_FLAGS_DEBUG}")
set( CMAKE_C_FLAGS_DEBUGULT "${CMAKE_C_FLAGS_DEBUG}")
set( CMAKE_EXE_LINKER_FLAGS_DEBUGULT "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
set( CMAKE_SHARED_LINKER_FLAGS_DEBUGULT "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
set( CMAKE_MODULE_LINKER_FLAGS_DEBUGULT "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")

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

if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") # GCC specific flags (somewhat recognized by ICC
add_definitions(-mavx)
add_definitions(-mavx2)
add_definitions(-mfma)
add_definitions(-ffast-math)
endif()

endif(UNIX)

# Define to distinguish among release and debug builds
if( NOT (CMAKE_BUILD_TYPE STREQUAL "Release") )
add_definitions(-DDEBUG)    
endif( NOT (CMAKE_BUILD_TYPE STREQUAL "Release") )

if( CMAKE_BUILD_TYPE STREQUAL "Release")
    if(UNIX)
        CHECK_CXX_COMPILER_FLAG("-march=core-avx2" COMPILER_SUPPORTS_ICC_AVX2)
        if(COMPILER_SUPPORTS_ICC_AVX2)
            add_definitions(-march=core-avx2)
        endif()
        CHECK_CXX_COMPILER_FLAG("-no-fma" COMPILER_SUPPORTS_ICC_NO_FMA)
        if(COMPILER_SUPPORTS_ICC_NO_FMA)
            add_definitions(-no-fma)
        endif()
        CHECK_CXX_COMPILER_FLAG("-fp-model fast" COMPILER_SUPPORTS_ICC_FP_MODEL_FAST)
        if(COMPILER_SUPPORTS_ICC_FP_MODEL_FAST)
            add_definitions(-fp-model fast)
        endif()
        CHECK_CXX_COMPILER_FLAG("-ipo" COMPILER_SUPPORTS_ICC_IPO)
        if(COMPILER_SUPPORTS_ICC_IPO)
            add_definitions(-ipo)
        endif()
    else()
        # windows
        CHECK_CXX_COMPILER_FLAG("/Ot" COMPILER_SUPPORTS_FAVORSPEED)
        CHECK_CXX_COMPILER_FLAG("/Oi" COMPILER_SUPPORTS_INTRINSIC)
        CHECK_CXX_COMPILER_FLAG("/Qipo" COMPILER_SUPPORTS_INTEROPT)
        CHECK_CXX_COMPILER_FLAG("/GL" COMPILER_SUPPORTS_WHOLEOPT)
        CHECK_CXX_COMPILER_FLAG("/GS-" COMPILER_SUPPORTS_NOSECCHECK)
        if(COMPILER_SUPPORTS_INTEROPT)
            add_definitions(/Qipo)
        endif()
        if(COMPILER_SUPPORTS_WHOLEOPT)
            add_definitions(/GL)
        endif()
        if(COMPILER_SUPPORTS_FAVORSPEED)
            add_definitions(/Ot)
        endif()
        if(COMPILER_SUPPORTS_INTRINSIC)
            add_definitions(/Oi)
        endif()
        if(COMPILER_SUPPORTS_NOSECCHECK)
            add_definitions(/GS-)
        endif()
    endif(UNIX)
    # 
endif()

# Set as advanced variables in GUI
mark_as_advanced(
    CMAKE_CXX_FLAGS_DEBUGULT
    CMAKE_C_FLAGS_DEBUGULT
    CMAKE_EXE_LINKER_FLAGS_DEBUGULT
    CMAKE_SHARED_LINKER_FLAGS_DEBUGULT
    CMAKE_MODULE_LINKER_FLAGS_DEBUGULT  )

# Set compiler flags and options. 
# For now it is default.
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Commands to output information to the console
# Useful for displaying errors, warnings, and debugging
message ("cxx Flags: " ${CMAKE_CXX_FLAGS})
message("${CMAKE_BUILD_TYPE}")

# Sub-directories where more CMakeLists.txt exist (more vcprojs)
add_subdirectory(devices/device_cpu)
add_subdirectory(devices/device_gpu)
add_subdirectory(node_runtime)
#
# Support for clang_complete (needed for easier navigation in Vim/Emacs and others supporting clang_complete)
include(./cmake/copy_clang_complete.cmake )

# Generate ULT projects only in DebugULT
if( CMAKE_BUILD_TYPE STREQUAL "DebugULT")
    # variables evening out the unix/windows name differences
    if(UNIX)
        set(ULTS_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        set(SDE_BINARY_NAME sde64)  #We support 64bit builds only
    else()
        set(ULTS_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE})
        set(SDE_BINARY_NAME sde)  
    endif()

    set(GFLAGS_BINARY_NAME gflags) 
    
    if(RUN_ULTS_OFFLINE)
        message("---> Offline testing enabled!")
        enable_testing()
    endif()

    add_subdirectory(gtest)
    add_subdirectory(unit_tests/ult_cpu)
    # add_subdirectory(unit_tests/ult_cpu_int16)
    add_subdirectory(unit_tests/ult_gpu)
    add_subdirectory(unit_tests/ult_runtime)
 endif()
