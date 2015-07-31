@REM Copyright (c) 2014, Intel Corporation
@REM 
@REM Redistribution and use in source and binary forms, with or without
@REM modification, are permitted provided that the following conditions are met:
@REM 
@REM     * Redistributions of source code must retain the above copyright notice,
@REM       this list of conditions and the following disclaimer.
@REM     * Redistributions in binary form must reproduce the above copyright
@REM       notice, this list of conditions and the following disclaimer in the
@REM       documentation and/or other materials provided with the distribution.
@REM     * Neither the name of Intel Corporation nor the names of its contributors
@REM       may be used to endorse or promote products derived from this software
@REM       without specific prior written permission.
@REM 
@REM THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
@REM AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
@REM IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
@REM DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
@REM FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
@REM DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
@REM SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
@REM CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
@REM OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
@REM OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@echo off
echo Creating Microsoft Visual Studio 2013 x64 files...

REM Following environment variables can customize build system:
REM USE_SDE_EMULATION -- Whether to use SDE emulation. Possible values: on, off  (lack of variable is equal to off)
REM RUN_ULTS_OFFLINE --  whether to separate ULTs execution from building process. Possible values: on, off (lack of variable is equal to off)
REM 
REM Examples:
REM set RUN_ULTS_OFFLINE=on & create_msvc_mscc.sh       

cmake -G"Visual Studio 12 2013 Win64" -B".\MSVC\Release" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=%RUN_ULTS_OFFLINE% -DUSE_SDE_EMULATION:BOOL=%USE_SDE_EMULATION% -DCMAKE_BUILD_TYPE:STRING="Release" -DCMAKE_CONFIGURATION_TYPES:STRING="Release" -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON -H"."
cmake -G"Visual Studio 12 2013 Win64" -B".\MSVC\Debug" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=%RUN_ULTS_OFFLINE% -DUSE_SDE_EMULATION:BOOL=%USE_SDE_EMULATION% -DCMAKE_BUILD_TYPE:STRING="Debug" -DCMAKE_CONFIGURATION_TYPES:STRING="Debug" -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON -H"."
cmake -G"Visual Studio 12 2013 Win64" -B".\MSVC\DebugULT" --no-warn-unused-cli -DRUN_ULTS_OFFLINE:BOOL=%RUN_ULTS_OFFLINE% -DUSE_SDE_EMULATION:BOOL=%USE_SDE_EMULATION% -DCMAKE_BUILD_TYPE:STRING="DebugULT" -DCMAKE_CONFIGURATION_TYPES:STRING="DebugULT" -DCMAKE_SUPPRESS_REGENERATION:BOOL=ON -H"."

echo Done.
pause
