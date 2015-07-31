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

#include "test_common_tools.h"
#include <random>
#include <limits>



//--------------RANDOM DATA GENERATOR-------------------//
void nn_data_populate(
        nn::data<float>* in_out,
        bool const_val,
        float const_val_value,
        float min_val,
        float max_val) {

    std::mt19937 generator(1);
    std::uniform_real_distribution<double> distribution(min_val,max_val);

    auto buffer_ptr = reinterpret_cast<float*> (in_out->buffer);
    uint32_t element_nr = in_out->count();

    while(element_nr--)
        *buffer_ptr++ = (const_val) ? const_val_value : distribution(generator);
};

    void nn_data_populate(
        nn::data<float>* in_out,
        float const_val) {

        return nn_data_populate(in_out,true,const_val);
    }

    void nn_data_populate(
        nn::data<float>* in_out,
        float min_val,
        float max_val) {

        return nn_data_populate(in_out,false,0,min_val,max_val);
    }
