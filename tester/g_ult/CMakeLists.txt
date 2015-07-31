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



# Generate ULT projects only in DebugULT
if( CMAKE_BUILD_TYPE STREQUAL "DebugULT")
    # Setup of DebugULT configuration(flags same as Debug)
    set(CMAKE_CXX_FLAGS_DEBUGULT "${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_C_FLAGS_DEBUGULT "${CMAKE_C_FLAGS_DEBUG}")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUGULT "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUGULT "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUGULT "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")

    # Set as advanced variables in GUI
    mark_as_advanced(
        CMAKE_CXX_FLAGS_DEBUGULT
        CMAKE_C_FLAGS_DEBUGULT
        CMAKE_EXE_LINKER_FLAGS_DEBUGULT
        CMAKE_SHARED_LINKER_FLAGS_DEBUGULT
        CMAKE_MODULE_LINKER_FLAGS_DEBUGULT)
    # variables evening out the unix/windows name differences

    message (STATUS "ULTS_WORKING_DIR ...................... " ${ULTS_WORKING_DIR})
    message (STATUS "ULTS_RUN_DIR ...................... " ${ULTS_RUN_DIR})

    set(GFLAGS_BINARY_NAME gflags) 
    
    if(RUN_ULTS_OFFLINE)
        message("---> Offline testing enabled!")
        enable_testing()
    endif()
    add_subdirectory(gtest)
    add_subdirectory(unit_tests/cpu)
    add_subdirectory(unit_tests/gpu)
 endif()
