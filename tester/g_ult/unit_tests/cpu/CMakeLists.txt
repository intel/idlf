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

# Sources for test files

file (GLOB TEST_CASES_SRC
      "test_cases/*.cpp")
      
# Main source file
set  (MAIN_SRC
      "main.cpp"
      "naive_implementations.cpp"
      "forward_convolve_simmplifying_wrapper.cpp"
      )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("" FILES ${MAIN_SRC})
source_group("test_cases" FILES ${TEST_CASES_SRC})

# Properties->C/C++->General->Additional Include Directories
include_directories ("${PROJECT_SOURCE_DIR}/tester/g_ult/gtest"
                     "${PROJECT_SOURCE_DIR}/tester/g_ult/gtest/include"
                     "${PROJECT_SOURCE_DIR}/tester/g_ult/unit_tests"
                     "${PROJECT_SOURCE_DIR}/device/"
                     )

# Create exe file from sources.
set (ULT_EXE_NAME "ult_cpu")
add_executable(${ULT_EXE_NAME} ${MAIN_SRC} 
                       ${TEST_CASES_SRC})

# To use SDE it has to be installed and its root dir needs to be
# added to PATH env variable.Also USE_SDE_EMULATION should be set
# to ON (see scripts for generating build system)
if(USE_SDE_EMULATION)
find_program(SDE_EXE_DIR ${SDE_BINARY_NAME})
message("SDE: ${SDE_EXE_DIR}")
endif()

if((SDE_EXE_DIR STREQUAL "SDE_EXE_DIR-NOTFOUND") OR (NOT USE_SDE_EMULATION))
set(TEST_COMMAND ${ULTS_RUN_DIR}/${ULT_EXE_NAME})
else()
set(TEST_COMMAND ${SDE_EXE_DIR} -bdw -- ${ULTS_RUN_DIR}/${ULT_EXE_NAME})
endif()

set_target_properties( ${ULT_EXE_NAME} 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${ULTS_RUN_DIR}"
)

set_target_properties( ${ULT_EXE_NAME}
                       PROPERTIES 
                       FOLDER tester/g_ult
                     )

set_target_properties(${ULT_EXE_NAME} 
                      PROPERTIES 
                      PROJECT_LABEL cpu) 

if(RUN_ULTS_OFFLINE)
    add_test(${ULT_EXE_NAME} ${TEST_COMMAND})
    message (STATUS "add_test ...................... " ${ULT_EXE_NAME} ${TEST_COMMAND})
else()
    if(NOT UNIX)
        find_program(GFLAGS_EXE_DIR ${GFLAGS_BINARY_NAME})
        message("GFLAGS: ${GFLAGS_EXE_DIR}")

        if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            add_custom_command(TARGET ${ULT_EXE_NAME}
                           POST_BUILD
                           WORKING_DIRECTORY ${ULTS_RUN_DIR}
                           COMMAND ${GFLAGS_EXE_DIR} /i ${ULT_EXE_NAME}.exe +ust +hpa )
        endif()
    endif()

    if((SDE_EXE_DIR STREQUAL "SDE_EXE_DIR-NOTFOUND") OR (NOT USE_SDE_EMULATION))
        add_custom_command(TARGET ${ULT_EXE_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${ULTS_RUN_DIR}
                       COMMAND ${ULT_EXE_NAME} )
    else()
        add_custom_command(TARGET ${ULT_EXE_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${ULTS_RUN_DIR}
                       COMMAND ${ULT_EXE_NAME} )
    endif()
    
    if(NOT UNIX)
        if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            add_custom_command(TARGET ${ULT_EXE_NAME}
                           POST_BUILD
                           WORKING_DIRECTORY ${ULTS_RUN_DIR}
                           COMMAND ${GFLAGS_EXE_DIR} /i ${ULT_EXE_NAME}.exe -ust -hpa )
        endif()
     endif()
endif()
# Set library dependencies
target_link_libraries(${ULT_EXE_NAME} gtest device_cpu)
