/*
Copyright (c) 2014, Intel Corporation

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

#pragma once

#include <algorithm>
#include <chrono>
#include <string>

class C_time_control
{
private:
    std::chrono::high_resolution_clock::time_point  time_tick;
    std::chrono::high_resolution_clock::time_point  time_tock;
    uint64_t                                        time_diff;
    uint64_t                                        clocks_tick;
    uint64_t                                        clocks_tock;
    uint64_t                                        clocks_diff;

public:
    C_time_control() { tick(); };
    ~C_time_control();

    void tick() {
        time_tick   = std::chrono::high_resolution_clock::now();
        clocks_tick = __rdtsc();
    };

    void tock() {
        clocks_tock = __rdtsc();
        time_tock   = std::chrono::high_resolution_clock::now();
        clocks_diff= clocks_tock - clocks_tick;
        time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(time_tock - time_tick).count();
    };

    uint64_t     get_time_diff() { return time_diff; };
    uint64_t     get_clocks_diff() { return clocks_diff; };

    std::string time_diff_string() { return time_diff_string(time_diff); };
    std::string clocks_diff_string() { return clocks_diff_string(clocks_diff); };

    static std::string time_diff_string(uint64_t);
    static std::string clocks_diff_string(uint64_t);
};

