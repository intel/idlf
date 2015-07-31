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

#include "gtest/gtest.h"

#include "device/common/nn_workload_data.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "device/cpu/core/fixedpoint/layer_convert_float_to_int16_fixedpoint_avx2.h"

#include <memory>

///////////////////////////////////////////////////////////////////////////////////////////////////
namespace {
    void test_setup(nn_device_description_t &device_description, nn_device_interface_0_t &device_interface_0) {
        // load device & validate it has 0 as a starting interface version
        nn_device_load(&device_description);
        EXPECT_EQ(device_description.version_first, 0);
        // open interface 0
        EXPECT_EQ(0, nn_device_interface_open(0, &device_interface_0));
    }

    void test_teardown(nn_device_description_t &device_description, nn_device_interface_0_t &device_interface_0) {
        // close interface 0
        EXPECT_EQ(0, nn_device_interface_close(&device_interface_0));
        // unload device
        EXPECT_EQ(0, nn_device_unload());
    }
}

TEST(cpu_int16_convert_float_to_int16_fixedpoint, convert_RGB_to_ZRGB) {
    const uint32_t width = 17, height = 7;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    nn_workload_data_layout_t input_layout = nn::workload_data<float>::layout.zxynpq;

    nn_workload_data_layout_t output_layout = nn::workload_data<int16_t>::layout.pxyznq;

    nn_workload_data_coords_t input_coords = {
        1, // batch size
        width,
        height,
        3,
        1,
        1,
    };

    nn_workload_data_coords_t output_coords = {
        1, // batch size
        width,
        height,
        1,
        4,
        1,
    };

    std::unique_ptr<nn::workload_data<float>> input_data(new nn::workload_data<float>(input_coords, input_layout));
    for (int i = 0; i < width * height * 3; ++i)
        reinterpret_cast<float *>(input_data->parent->data_buffer)[i] = i;

    std::unique_ptr<nn::workload_data<std::int16_t>> output_data(new nn::workload_data<std::int16_t>(output_coords, output_layout));

    std::unique_ptr<nn_workload_item> input_item(new nn_workload_item());
    input_item->output.push_back(input_data.get());

    std::unique_ptr<nn_workload_item> work_item(new nn_workload_item());
    work_item->type = NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT;
    auto &arguments = work_item->arguments.forward_convert_float_to_fixedpoint;
    arguments.output_fraction = 8;

    work_item->input.push_back({ input_item.get(), 0 });
    work_item->output.push_back(output_data.get());

    // Create primitive
    work_item->primitive = new int16_fixedpoint::convert_float_to_int16(
        4,
        width,
        height,
        1,
        0,
        0,
        0,
        0,
        arguments.output_fraction,
        reinterpret_cast<nn_device_internal*>(device_interface_0.device));

    work_item->output[0] = static_cast<int16_fixedpoint::convert_float_to_int16 *>(work_item->primitive)->create_outputs(false)[0];

    int16_fixedpoint::run_convert_float_to_int16_fp_work_item(work_item.get(), reinterpret_cast<nn_device_internal*>(device_interface_0.device));

    auto float2int16 = [&arguments](float in) {
        auto scaled = in * (1 << arguments.output_fraction);
        if (scaled < INT16_MIN)
            scaled = INT16_MIN;
        if (scaled > INT16_MAX)
            scaled = INT16_MAX;
        return static_cast<std::int16_t>(scaled);
    };

    auto check_result = [&](float *in, std::int16_t *out) {
        for (int i = 0; i < width * height; ++i) {
            if (out[i * 4 + 0] != float2int16(in[i * 3 + 0]))
                return false;
            if (out[i * 4 + 1] != float2int16(in[i * 3 + 1]))
                return false;
            if (out[i * 4 + 2] != float2int16(in[i * 3 + 2]))
                return false;
            if (out[i * 4 + 3] != 0)
                return false;
        }
        return true;
    };

    test_teardown(device_description, device_interface_0);

    EXPECT_TRUE(check_result(reinterpret_cast<float *>(input_data->parent->data_buffer),
        reinterpret_cast<std::int16_t *>(work_item->output[0]->parent->data_buffer)));
}
