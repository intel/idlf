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

add_subdirectory(g_ult)

set (TESTER_APP_NAME "tester")

project (${TESTER_APP_NAME})


set (LOCAL_COMMON_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/common)
set (COMMON_SRC_DIR  ${CMAKE_SOURCE_DIR}/common)
set (TESTS_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set (WORKFLOWS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/workflows)

include_directories(${CMAKE_SOURCE_DIR})

file (GLOB MAIN_SRC
       main.cpp
       disabled_tests.*
      )

file (GLOB COMMON_SRC
     ${COMMON_SRC_DIR}/*.cpp
     ${COMMON_SRC_DIR}/*.h
     ${LOCAL_COMMON_SRC_DIR}/*.cpp
     ${LOCAL_COMMON_SRC_DIR}/*.h
     )

file (GLOB TESTS_SRC
     ${TESTS_SRC_DIR}/*.cpp
     ${TESTS_SRC_DIR}/*.h
     )

file (GLOB WORKFLOWS_SRC
     ${WORKFLOWS_SRC_DIR}/*.cpp
     )
file (GLOB WORKLOAD_FOR_TEST_SRC
     ${CMAKE_SOURCE_DIR}/device/common/*.cpp
     ${CMAKE_SOURCE_DIR}/device/common/*.h
     )


add_executable(${TESTER_APP_NAME}
                 ${MAIN_SRC}
                 ${COMMON_SRC}
                 ${TESTS_SRC}
                 ${WORKFLOWS_SRC}
                 ${WORKLOAD_FOR_TEST_SRC}
              )

set_target_properties( ${TESTER_APP_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tester/${COMPILER_STR}${BUILD_TYPE}"
)

set_target_properties( ${TESTER_APP_NAME}
                       PROPERTIES
                       FOLDER tester
                     )

set_target_properties(${TESTER_APP_NAME}
                      PROPERTIES
                      PROJECT_LABEL ${TESTER_APP_NAME})

source_group (
               "tests"
               FILES ${TESTS_SRC}
             )
source_group (
               "main"
               FILES ${MAIN_SRC}
             )
source_group (
               "common"
               FILES ${COMMON_SRC}
             )

source_group (
               "tests\\workflows_for_tests"
               FILES ${WORKFLOWS_SRC}
             )
source_group (
               "workload"
               FILES ${WORKLOAD_FOR_TEST_SRC}
             )

set (POST_BUILD_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tester/${COMPILER_STR}${BUILD_TYPE})

target_link_libraries(${TESTER_APP_NAME} ${FreeImage_LIBRARIES} asmjit)

if (UNIX)
    target_link_libraries(${TESTER_APP_NAME} dl)
    target_link_libraries(${TESTER_APP_NAME} pthread)

endif()
if (NOT UNIX)
  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_custom_command(TARGET ${TESTER_APP_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${POST_BUILD_WORKING_DIRECTORY}/${CMAKE_BUILD_TYPE}
                       COMMAND "${CMAKE_COMMAND}" -E copy ${TESTER_APP_NAME}.exe ${RUNTIME_BIN_DIRECTORY}
                       )
  else()
    add_custom_command(TARGET ${TESTER_APP_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${POST_BUILD_WORKING_DIRECTORY}/${CMAKE_BUILD_TYPE}
                       COMMAND "${CMAKE_COMMAND}" -E copy ${TESTER_APP_NAME}.exe ${RUNTIME_BIN_DIRECTORY}
                       COMMAND "${CMAKE_COMMAND}" -E copy ${TESTER_APP_NAME}.pdb ${RUNTIME_BIN_DIRECTORY}
                       )
  endif()
endif()
