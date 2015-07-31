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
#include "time_control.h"
#include <sstream>
#include <iomanip>

C_time_control::~C_time_control() {

}

std::string C_time_control::time_diff_string(uint64_t t_diff) {
    std::ostringstream temp;
    if(t_diff>0) {
        double t_d = static_cast<double>(t_diff);
        std::string units[]={ "ns", "us", "ms", "s" };
        uint8_t     index=0;

        while(t_d>1000 && index < 3) {
            t_d/=1000;
            ++index;
        };
    temp << std::setprecision(3) << std::fixed << t_d << " " + units[index];
    }
    return temp.str();
}
std::string C_time_control::clocks_diff_string(uint64_t c_diff) {
    std::string        cds;
    std::ostringstream rest;

    while(c_diff>999) {
        rest.str("");
        rest <<" "<< std::setfill('0') << std::setw(3) <<  c_diff % 1000;
        c_diff/=1000;
        cds = rest.str() + cds;
    }
    if(c_diff>0) {
        rest.str("");
        rest<<c_diff;
        cds = rest.str() + cds;
    }
    return cds + " ticks";
}
