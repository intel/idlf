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

#include "helper_zxyn_f32.h"
#include "device/cpu/api_internal/data_helper.h"

namespace layer {
namespace helper_zxyn_f32 {

std::vector<nn_workload_data_t *> primitive_zxyn_f32_base::create_inputs(bool allocate_delta) {
    return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(
        device,
        static_cast<uint32_t>(get_required_input_w()),
        static_cast<uint32_t>(get_required_input_h()),
        static_cast<uint32_t>(input_size_z),
        static_cast<uint32_t>(batch_size),
        0,
        0,
        0,
        0,
        allocate_delta) };
}

std::vector<nn_workload_data_t *> primitive_zxyn_f32_base::create_outputs(bool allocate_delta) {
    return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(device,
        static_cast<uint32_t>(output_size_x),
        static_cast<uint32_t>(output_size_y),
        static_cast<uint32_t>(output_size_z),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(output_padding_left),
        static_cast<uint32_t>(output_padding_right),
        static_cast<uint32_t>(output_padding_top),
        static_cast<uint32_t>(output_padding_bottom),
        allocate_delta) };
}

nn::workload_data<> *primitive_zxyn_f32_base::map_input(const nn::data<float, 4> &input) {
     if (input.size[3] != batch_size)
        throw std::invalid_argument("input batch size doesn't match");

    if (input.size[0] != input_size_z)
        throw std::invalid_argument("number of input feature maps doesn't match");

    if (input.size[1] != get_required_input_w())
        throw std::invalid_argument("input width doesn't match");

    if (input.size[2] != get_required_input_h())
        throw std::invalid_argument("input height doesn't match");

    nn_workload_data_coords_t size = {static_cast<uint32_t>(input.size[3]), // n
                                      static_cast<uint32_t>(input.size[1]), // x
                                      static_cast<uint32_t>(input.size[2]), // y
                                      static_cast<uint32_t>(input.size[0]), // z
                                      1,
                                      1};

    auto buffer = new nn::workload_data<>(
        NN_WORKLOAD_DATA_TAG_ZXYN, input.buffer, size, nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::layout);

    return buffer;
}

}
}
