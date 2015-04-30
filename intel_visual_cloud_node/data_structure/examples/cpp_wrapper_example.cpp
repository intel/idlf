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

#include "nn_workload_data_cpp_wrapper.h"
#include <iostream>

int main_cpp_wrapper_example()
{
    nn_workload_data_coords_t nn_coords = { 4, 4, 4, 4, 4, 4 };

    {
        // FLOAT32
        const float value = 3.14f;
        nn_workload_data_layout_t layout = {
            { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
            { 0, 0, 0, 0, 0, 0 }, // alignment
            { 0, 1, 2, 3, 4, 5 }, // ordering
            NN_DATATYPE_FLOAT
        };

        std::cout << "\nFloat C++ wrapper example" << std::endl;

        nn::nn_workload_data_t<float> nndata(nn_coords, layout);

        nndata(1, 1, 2, 3, 2, 2) = value;

        std::cout << "Value written: " << value << " value read: " << nndata(1, 1, 2, 3, 2, 2)() << std::endl;

        {
            nn_workload_data_coords_t nn_view_begin = { 1, 1, 1, 1, 1, 1 };
            nn_workload_data_coords_t nn_view_end = { 3, 3, 3, 3, 3, 3 };

            nn::nn_workload_data_t<float> nnview(nndata, nn_view_begin, nn_view_end);
            std::cout << "Value read from the view: " << nnview(0, 0, 1, 2, 1, 1)() << std::endl;
        }
    }
    
    {
        // INT16
        const int16_t value = 1234;
        nn_workload_data_layout_t layout = {
            { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
            { 0, 0, 0, 0, 0, 0 }, // alignment
            { 0, 1, 2, 3, 4, 5 }, // ordering
            NN_DATATYPE_INT16
        };

        std::cout << "\nInt16 C++ wrapper example" << std::endl;

        nn::nn_workload_data_t<int16_t> nndata(nn_coords, layout);

        nndata(1, 1, 2, 3, 2, 2) = value;

        std::cout << "Value written: " << value << " value read: " << nndata(1, 1, 2, 3, 2, 2)() << std::endl;

        {
            nn_workload_data_coords_t nn_view_begin = { 1, 1, 1, 1, 1, 1 };
            nn_workload_data_coords_t nn_view_end = { 3, 3, 3, 3, 3, 3 };

            nn::nn_workload_data_t<int16_t> nnview(nndata, nn_view_begin, nn_view_end);
            std::cout << "Value read from the view: " << nnview(0, 0, 1, 2, 1, 1)() << std::endl;
        }
    }

    return 0;
}