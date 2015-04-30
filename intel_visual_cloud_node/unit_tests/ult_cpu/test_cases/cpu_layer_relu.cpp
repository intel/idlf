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
#include "../../devices/api/nn_primitives_api_0.h"

#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string>

void relu_reference(const nn::data<float, 4> &input, nn::data<float, 4> &output)
{
    for (int index = 0; index < output.count(); ++index)
    {
        ((float *)(output.buffer))[index] = std::max(0.0f, ((float *)(input.buffer))[index]);
    }
}

nn::data<float, 4> create_populated_input(size_t data_x, size_t data_y, size_t data_z, size_t batch_size)
{
    nn::data<float, 4> input(data_z, data_x, data_y, batch_size);
    for (int i = 0; i < input.count(); ++i)
    {
        ((float *)(input.buffer))[i] = i * (i % 2 == 0 ? -1 : 1); // change sign to test relu
    }

    return input;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static bool ult_perform_test(
    size_t data_x,
    size_t data_y,
    size_t data_z,
    size_t batch_size)
{
    bool passed = true;
    nn_device_primitives_description_t device_primitives_description;
    nn_device_get_primitives_description(&device_primitives_description);
    assert(device_primitives_description.version_first <= 0);
    assert(device_primitives_description.version_last >= 0);

    nn_primitives_0_t primitives;
    nn_device_get_primitives(0, &primitives);
    nn_device_t *device = primitives.create_device_with_thread_count(0, nullptr);

    // prepare data
    auto input = create_populated_input(data_x, data_y, data_z, batch_size);

    // calculate reference output
    nn::data<float, 4> output_ref(data_z, data_x, data_y, batch_size);
    nn::data<float, 4> output(data_z, data_x, data_y, batch_size);
    relu_reference(input, output_ref);

    // init primitive handle
    nn_primitive_handle_t primitive = primitives.relu_f32->create_handle(device,
                                                                         data_x,
                                                                         data_y,
                                                                         data_z,
                                                                         batch_size,
                                                                         nullptr);

    // prepare buffers
    nn_opaque_data_t *input_internal, *output_internal;
    input_internal = primitives.relu_f32->create_input(primitive, &input, nullptr);
    output_internal = primitives.relu_f32->create_output(primitive, nullptr);

    // execute relu
    nn_event_t relu = primitives.relu_f32->forward_async(primitive, input_internal, output_internal, 0, nullptr, nullptr);

    // extract output
    nn_event_t output_ready = primitives.relu_f32->copy_output_async(primitive, &output, output_internal, 1, &relu, nullptr);
    primitives.wait(1, &output_ready);

    // compare results
    for (size_t output_x = 0; output_x < data_x; ++output_x)
    {
        for (size_t output_y = 0; output_y < data_y; ++output_y)
        {
            for (size_t output_z = 0; output_z < data_z; ++output_z)
            {
                auto &output_val = output.at(output_z, output_x, output_y, 0);
                auto &output_ref_val = output_ref.at(output_z, output_x, output_y, 0);
                auto error = fabs((output_ref_val - output_val) / output_ref_val);
                if (error > 0.10f)
                    passed = false;
            }
        }
    }

    // cleanup
    primitives.delete_event(output_ready);
    primitives.delete_event(relu);
    primitives.delete_opaque_data(input_internal);
    primitives.delete_opaque_data(output_internal);
    primitives.delete_device(device);

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_relu, cpu_relu_test)
{
    EXPECT_EQ(true, ult_perform_test(3, 3, 32, 1)); /* x, y, x, batch */
}