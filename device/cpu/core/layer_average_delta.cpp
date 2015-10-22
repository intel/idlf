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

#include "layer_average_delta.h"

namespace layer
{
    void run_average_delta(nn_workload_item *const item)
    {
        assert(item->input.size() == item->output.size());

        for (uint32_t parameter = 0; parameter < item->input.size(); ++parameter)
        {
            auto input = nn::workload_data_cast<nn::layout_f32>(item->input[parameter].get_data_view());
            auto output = nn::workload_data_cast<nn::layout_f32>(item->output[parameter]);

            auto input_batch = input->parent->lengths.t[NN_DATA_COORD_n];
            auto output_batch = output->parent->lengths.t[NN_DATA_COORD_n];

            if (input_batch == output_batch)
                nn_workload_data_copy(output, input);
            else
            {
                for (uint32_t x = 0; x < output->parent->lengths.t[NN_DATA_COORD_x]; ++x)
                    for (uint32_t y = 0; y < output->parent->lengths.t[NN_DATA_COORD_y]; ++y)
                        for (uint32_t z = 0; z < output->parent->lengths.t[NN_DATA_COORD_z]; ++z)
                            for (uint32_t p = 0; p < output->parent->lengths.t[NN_DATA_COORD_p]; ++p)
                                for (uint32_t q = 0; q < output->parent->lengths.t[NN_DATA_COORD_q]; ++q)
                                {
                                    float acc = 0.0f;
                                    for (uint32_t n = 0; n < input_batch; ++n)
                                        acc += input->at(n, x, y, z, p, q);

                                    acc /= static_cast<float>(input_batch);

                                    (*output)(0, x, y, z, p, q) = acc;
                                }
            }
        }
    }
}