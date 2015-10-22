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

# Sources for test files
set (TEST_CASES_SRC
      "test_cases/gpu_device_workflow_interface_0_functions.cpp"
      )
      
# Main source file
set  (MAIN_SRC
      "main.cpp"
      "common.h"
      "common.cpp")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("" FILES ${MAIN_SRC})
source_group("test_cases" FILES ${TEST_CASES_SRC})


# Get the list of directories where libs could be stored
file(GLOB OPENCL_LIB_DIRS $ENV{INTELOCLSDKROOT}/../**/lib64/)

# Order of searching is that first SDK is checked for OpenCL headers/libs then 
# headers from project are taken and libs from hardcoded system locations
find_path(OPENCL_INCLUDE_DIRS NAMES CL/cl.hpp HINTS $ENV{INTELOCLSDKROOT}/include PATHS ../../ )
find_library(OPENCL_LIBRARIES NAMES OpenCL OpenCL.lib HINTS $ENV{INTELOCLSDKROOT}/lib/x64
 ${OPENCL_LIB_DIRS} PATHS /lib64 /usr/lib )

# Properties->C/C++->General->Additional Include Directories
include_directories ("${PROJECT_SOURCE_DIR}/tester/g_ult/gtest" "${PROJECT_SOURCE_DIR}/tester/g_ult/gtest/include" "${OPENCL_INCLUDE_DIRS}")

# Create exe file from sources.
set (ULT_EXE_NAME "ult_gpu")
add_executable(${ULT_EXE_NAME} ${MAIN_SRC} 
                       ${TEST_CASES_SRC})


set_target_properties( ${ULT_EXE_NAME} 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${ULTS_WORKING_DIR}"
)

set_target_properties( ${ULT_EXE_NAME}
                       PROPERTIES 
                       FOLDER tester/g_ult
                     )

set_target_properties(${ULT_EXE_NAME} 
                      PROPERTIES 
                      PROJECT_LABEL gpu) 

#if(RUN_ULTS_OFFLINE)
    #add_test(${ULT_EXE_NAME} ${ULTS_RUN_DIR}/${ULT_EXE_NAME})
    #message (STATUS "add_test ...................... " ${ULT_EXE_NAME} ${ULTS_RUN_DIR}/${ULT_EXE_NAME})
#else()

    #if(NOT UNIX)
        #find_program(GFLAGS_EXE_DIR ${GFLAGS_BINARY_NAME})
        #message("GFLAGS: ${GFLAGS_EXE_DIR}")

        #if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            #add_custom_command(TARGET ${ULT_EXE_NAME}
                           #POST_BUILD
                           #WORKING_DIRECTORY ${ULTS_RUN_DIR}/
                           #COMMAND ${GFLAGS_EXE_DIR} /i ${ULT_EXE_NAME}.exe +ust +hpa )
        #endif()
    #endif()

    ## Add post-build test run event	
    #add_custom_command(TARGET ${ULT_EXE_NAME}
                       #POST_BUILD
                       #WORKING_DIRECTORY ${ULTS_RUN_DIR}
                       #COMMAND ${ULT_EXE_NAME} )
                       
    #if(NOT UNIX)
        #if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            #add_custom_command(TARGET ${ULT_EXE_NAME}
                           #POST_BUILD
                           #WORKING_DIRECTORY ${ULTS_RUN_DIR}
                           #COMMAND ${GFLAGS_EXE_DIR} /i ${ULT_EXE_NAME}.exe -ust -hpa )
        #endif()
     #endif()
#endif()

# Set library dependencies
target_link_libraries(${ULT_EXE_NAME} gtest device_gpu ${OPENCL_LIBRARIES})
