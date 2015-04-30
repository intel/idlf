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

#!/bin/bash
BUILD_DIR=${1:-"UnixMk_icc"}

echo Creating echo Creating Unix Makefiles with ICC...

/opt/intel/bin/compilervars.sh intel64

# Following environment variables can customize build system:
# USE_SDE_EMULATION -- Whether to use SDE emulation. Posible values: on, off  (lack of varable is equal to off)
# RUN_ULTS_OFFLINE --  whether to separate ULTs execution from building process. Possible values: on, off (lack of variable is equal to off)
# 
# Examples:
# RUN_ULTS_OFFLINE=on ./create_unixmax_icc.sh       # This will make ULTs running offline

CC=icc CXX=icpc cmake -G"Unix Makefiles" -B"./${BUILD_DIR}/Release" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Release" -DCMAKE_CONFIGURATION_TYPES:STRING="Release" -H"."
CC=icc CXX=icpc cmake -G"Unix Makefiles" -B"./${BUILD_DIR}/Debug" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -H"."
CC=icc CXX=icpc cmake -G"Unix Makefiles" -B"./${BUILD_DIR}/DebugULT" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=${RUN_ULTS_OFFLINE} -DUSE_SDE_EMULATION:BOOL=${USE_SDE_EMULATION} -DCMAKE_BUILD_TYPE:STRING="DebugULT" -DCMAKE_CONFIGURATION_TYPES:STRING="DebugULT" -H"."

echo Done.
