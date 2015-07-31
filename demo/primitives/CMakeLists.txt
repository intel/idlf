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

#sets for DEMO_PRIMITIVES --------------------------------------------------------------------------------------------

set (DEMO_NAME "primitives")
set (TARGET_NAME demo_${DEMO_NAME})

project (${DEMO_NAME})

file (GLOB PRIMITIVE_WORKLOADS_SRC
      primitives_workload.h
      primitives_workload_*.cpp)


add_executable(${TARGET_NAME} primitives_main.cpp ${COMMON_SRC} ${PRIMITIVE_WORKLOADS_SRC})

target_link_libraries(${TARGET_NAME} ${FreeImage_LIBRARIES})
if(UNIX)
  target_link_libraries(${TARGET_NAME} dl)
endif()
set_target_properties( ${TARGET_NAME} 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/demo/${DEMO_NAME}/${COMPILER_STR}${BUILD_TYPE}"
)
set_target_properties( ${TARGET_NAME}
                       PROPERTIES 
                       FOLDER demo
                     )

set_target_properties(${TARGET_NAME} 
                      PROPERTIES 
                      PROJECT_LABEL ${DEMO_NAME}) 

set (POST_BUILD_WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/demo/${DEMO_NAME}/${COMPILER_STR}/${CMAKE_BUILD_TYPE})

if (NOT UNIX)
  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${POST_BUILD_WORKING_DIRECTORY}
                       COMMAND "${CMAKE_COMMAND}" -E copy ${TARGET_NAME}.exe ${RUNTIME_BIN_DIRECTORY}
                       )
  else()
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${POST_BUILD_WORKING_DIRECTORY}
                       COMMAND "${CMAKE_COMMAND}" -E copy ${TARGET_NAME}.exe ${RUNTIME_BIN_DIRECTORY}
                       COMMAND "${CMAKE_COMMAND}" -E copy ${TARGET_NAME}.pdb ${RUNTIME_BIN_DIRECTORY}
                       )
  endif()
endif()



