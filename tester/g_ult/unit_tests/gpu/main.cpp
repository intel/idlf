/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "gtest/gtest.h"

int main( int argc, char* argv[ ] )
{
    int result;
    {
        ::testing::InitGoogleTest(&argc, argv);

#if _WIN32
        // Safety cleanup.
        system("where /q umdh && del pre_gpu.txt");
        system("where /q umdh && del post_gpu.txt");
        system("where /q umdh && del memdiff_gpu.txt");

        // Get first snapshot.
        system("where /q umdh && umdh -pn:ult_gpu.exe -f:pre_gpu.txt");
#endif

        result = RUN_ALL_TESTS();
    }

#if _WIN32
    // Get second snapshot.
    system("where /q umdh && umdh -pn:ult_gpu.exe -f:post_gpu.txt");

    // Prepare memory diff.
    system("where /q umdh && umdh pre_gpu.txt post_gpu.txt -f:memdiff_gpu.txt");

    // Cleanup.
    system("where /q umdh && del pre_gpu.txt");
    system("where /q umdh && del post_gpu.txt");
#endif

    return result;
}
