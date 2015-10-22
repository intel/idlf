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
include(CheckCXXCompilerFlag)

#Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

message("=====================> ${CMAKE_BUILD_TYPE} <=====================")

set(PROJECT_NAME "intel_deep_learning_framework")

project (${PROJECT_NAME})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set (COMPILER_STR "gcc")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set (COMPILER_STR "icc")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set (COMPILER_STR "mscc")
endif()


set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set (RUNTIME_BIN_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set (COMMON_SRC_DIR  ${CMAKE_SOURCE_DIR}/common)

set (BUILD_TYPE "") 
if (UNIX)
    set (BUILD_TYPE "/${CMAKE_BUILD_TYPE}")
endif()

message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "CMAKE_SOURCE_DIR ...................... " ${CMAKE_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})

# export compilation flags for each source to help Vim/YCM have omnicompletition
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# To Enable C++11 gcc (Linux compiler) require additional flag to be added

add_definitions(-DXBYAK_NO_OP_NAMES)
add_definitions(-DUSE_XBYAK_CONVOLUTION)
add_definitions(-DXBYAK_USE_MMAP_ALLOCATOR)
if(UNIX)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if(COMPILER_SUPPORTS_CXX11)
    add_definitions(-std=c++11)
  elseif(COMPILER_SUPPORTS_CXX0X)
    add_definitions(-std=c++0x)
  endif()
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") # GCC specific flags
    add_definitions(-mavx)
    add_definitions(-mavx2)
    add_definitions(-mfma)
    add_definitions(-ffast-math)
  endif()
  add_definitions(-fPIC)
else()
    add_definitions(-FIciso646) 
endif()

if(UNIX)
  CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" COMPILER_SUPPORTS_VISIBILITY)
  if (COMPILER_SUPPORTS_VISIBILITY)
    add_definitions(-fvisibility=hidden)
  endif(COMPILER_SUPPORTS_VISIBILITY)
endif(UNIX)

if( CMAKE_BUILD_TYPE STREQUAL "DebugULT")
    # Setup of DebugULT configuration(flags same as Debug)
    set( CMAKE_CXX_FLAGS_DEBUGULT "${CMAKE_CXX_FLAGS_DEBUG}")
    set( CMAKE_C_FLAGS_DEBUGULT "${CMAKE_C_FLAGS_DEBUG}")
    set( CMAKE_EXE_LINKER_FLAGS_DEBUGULT "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
    set( CMAKE_SHARED_LINKER_FLAGS_DEBUGULT "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
    set( CMAKE_MODULE_LINKER_FLAGS_DEBUGULT "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
    if(UNIX)
        set(SDE_BINARY_NAME sde64)  #We support 64bit builds only
        set(ULTS_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tester/g_ult/${COMPILER_STR}/${CMAKE_BUILD_TYPE})
        set(ULTS_RUN_DIR ${ULTS_WORKING_DIR})
    else()
        set(SDE_BINARY_NAME sde)
        set(ULTS_WORKING_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tester/g_ult/${COMPILER_STR})
        set(ULTS_RUN_DIR ${ULTS_WORKING_DIR}/${CMAKE_BUILD_TYPE})
    endif()
endif()

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
        add_definitions(-march=native)
    else()
        # windows
        CHECK_CXX_COMPILER_FLAG("/Ot" COMPILER_SUPPORTS_FAVORSPEED)
        CHECK_CXX_COMPILER_FLAG("/Oi" COMPILER_SUPPORTS_INTRINSIC)
        CHECK_CXX_COMPILER_FLAG("/GS-" COMPILER_SUPPORTS_NOSECCHECK)
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
endif()

include_directories (${PROJECT_SOURCE_DIR}
                     ${PROJECT_SOURCE_DIR}/asmjit/src)

if(UNIX) # On Linux Freeimage provided by host OS is used
  find_package(FreeImage)
else()   # For windows we will used FreeImage binary stored at repo
  find_library(FreeImage_LIBRARIES FreeImage PATHS ${COMMON_SRC_DIR}/FreeImage64)
  set(FreeImage_INCLUDE_DIRS "${COMMON_SRC_DIR}FreeImage64")
endif()

# Sub-directories where more CMakeLists.txt exist
set(ASMJIT_STATIC TRUE)
add_subdirectory(asmjit)
add_subdirectory(device)
add_subdirectory(tester)
if (NOT (${CMAKE_BUILD_TYPE} STREQUAL "DebugULT"))
    add_subdirectory(demo)
endif()

