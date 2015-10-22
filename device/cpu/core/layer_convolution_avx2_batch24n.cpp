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

#include "device/common/nn_workload_data.h"
#include "device/common/nn_allocate.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_convolution_avx2_batch24n.h"
#include "helper_zxyn_f32.h"
#include "layer_convolution_avx2_forward.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <tuple>
#include <iostream>
#include "device/cpu/api_internal/data_helper.h"
#include "tester/g_ult/unit_tests/cpu/naive_implementations.h"
#include <cstddef>
#include <cstdint>
#include <iostream>


namespace layer
{
namespace
{
const uint32_t BATCH_BLOCK_LENGTH = BATCH_ACCEPTED_BLOCK;

uint32_t ceil_div(uint32_t arg, uint32_t div)
{
    return (arg + div - 1) / div;
}

} // namespace

template <typename T_Name>
ValueU64<T_Name> make(uint64_t val) { return ::make<T_Name, uint64_t>(val); }

void convolution_f32_batch24n::forward(const nn::workload_data<> *input_buffer,
                              const nn::workload_data<> *weights_buffer,
                              const nn::workload_data<> *bias_buffer,
                              nn::workload_data<> *output_buffer)
{
    if (prepared_for != std::make_tuple((float*)input_buffer->parent->data_buffer,
                                        (float*)output_buffer->parent->data_buffer,
                                        (float*)weights_buffer->parent->data_buffer,
                                        (float*)bias_buffer->parent->data_buffer))
        throw std::runtime_error("convolution forward called on different buffers than prepared for");

    convolution::forward::convolve_fst_threaded_batch(
        const_cast<nn_device_internal*>(device)->thread_pool,
        compiled_convolution,
        (float*)input_buffer->parent->data_buffer,
        (float*)output_buffer->parent->data_buffer,
        (float*)weights_buffer->parent->data_buffer,
        (float*)bias_buffer->parent->data_buffer,
        batch_size,
        kernel_info,
        in_full_dims,
        out_full_dims,
        stride_info,
        make<InputFeatsStart>(input_buffer->view_begin.t[NN_DATA_COORD_z]),
        make<OutputFeatsStart>(output_buffer->view_begin.t[NN_DATA_COORD_z]));
}

void convolution_f32_batch24n::prepare_forward(
    const nn::workload_data<> *input_buffer,
    const nn::workload_data<> *weights_buffer,
    const nn::workload_data<> *bias_buffer,
    nn::workload_data<> *output_buffer)
{
    using namespace convolution::forward;
    if (input_buffer->get_length(NN_DATA_COORD_n) != ceil_div(batch_size, BATCH_BLOCK_LENGTH))
        throw std::runtime_error("invalid input buffer length of n dimension");
    if (input_buffer->get_length(NN_DATA_COORD_z) != kernel_info.dims.feats)
        throw std::runtime_error("invalid input buffer length of z dimension");
    if (input_buffer->get_length(NN_DATA_COORD_p) != BATCH_BLOCK_LENGTH)
        throw std::runtime_error("invalid input buffer length of p dimension");
    if (input_buffer->get_length(NN_DATA_COORD_q) != 1)
        throw std::runtime_error("invalid input buffer length of q dimension");
    if (input_buffer->get_length(NN_DATA_COORD_x) < 1 + stride_info.stride.cols * (output_buffer->get_length(NN_DATA_COORD_x) - 1))
        throw std::runtime_error("invalid input buffer length of x dimension");
    if (input_buffer->get_length(NN_DATA_COORD_y) < 1 + stride_info.stride.rows * (output_buffer->get_length(NN_DATA_COORD_y) - 1))
        throw std::runtime_error("invalid input buffer length of y dimension");

    if (output_buffer->get_length(NN_DATA_COORD_n) != ceil_div(batch_size, BATCH_BLOCK_LENGTH))
        throw std::runtime_error("invalid output buffer length of n dimension");
    if (output_buffer->get_length(NN_DATA_COORD_z) != out_dims.feats)
        throw std::runtime_error("invalid output buffer length of z dimension");
    if (output_buffer->get_length(NN_DATA_COORD_x) != out_dims.width)
        throw std::runtime_error("invalid output buffer length of x dimension");
    if (output_buffer->get_length(NN_DATA_COORD_y) != out_dims.height)
        throw std::runtime_error("invalid output buffer length of y dimension");
    if (output_buffer->get_length(NN_DATA_COORD_p) != BATCH_BLOCK_LENGTH)
        throw std::runtime_error("invalid output buffer length of p dimension");
    if (output_buffer->get_length(NN_DATA_COORD_q) != 1)
        throw std::runtime_error("invalid output buffer length of q dimension");

    if (bias_buffer->get_length(NN_DATA_COORD_x) != out_dims.feats)
        throw std::runtime_error("invalid bias buffer length of x dimension");
    if (bias_buffer->get_length(NN_DATA_COORD_z) != 1u)
        throw std::runtime_error("invalid bias buffer length of z dimension");
    if (bias_buffer->get_length(NN_DATA_COORD_n) != 1u)
        throw std::runtime_error("invalid bias buffer length of n dimension");
    if (bias_buffer->get_length(NN_DATA_COORD_y) != 1u)
        throw std::runtime_error("invalid bias buffer length of y dimension");
    if (bias_buffer->get_length(NN_DATA_COORD_q) != 1u)
        throw std::runtime_error("invalid bias buffer length of q dimension");
    if (bias_buffer->get_length(NN_DATA_COORD_p) != 1u)
        throw std::runtime_error("invalid bias buffer length of p dimension");

    if (weights_buffer->get_length(NN_DATA_COORD_n) != 1)
        throw std::runtime_error("invalid weight buffer length of n dimension");
    if (weights_buffer->get_length(NN_DATA_COORD_x) != kernel_info.dims.width)
        throw std::runtime_error("invalid weight buffer length of x dimension");
    if (weights_buffer->get_length(NN_DATA_COORD_y) != kernel_info.dims.height)
        throw std::runtime_error("invalid weight buffer length of y dimension");
    if (weights_buffer->get_length(NN_DATA_COORD_z) != kernel_info.dims.feats)
        throw std::runtime_error("invalid weight buffer length of z dimension");
    if (weights_buffer->get_length(NN_DATA_COORD_p) != OUTPUT_ACCEPTED_MOD)
        throw std::runtime_error("invalid weight buffer length of p dimension");
    if (weights_buffer->get_length(NN_DATA_COORD_q) != ceil_div(out_dims.feats, OUTPUT_ACCEPTED_MOD))
        throw std::runtime_error("invalid weight buffer length of q dimension");

    if (weights_buffer->parent->lengths != weights_buffer->get_length())
        throw std::runtime_error("unexpected view on weight buffer");

    if (output_buffer->parent->lengths.t[NN_DATA_COORD_x] != output_buffer->get_length(NN_DATA_COORD_x))
        throw std::runtime_error("unexpected view on x dimension on output buffer");
    if (output_buffer->parent->lengths.t[NN_DATA_COORD_y] != output_buffer->get_length(NN_DATA_COORD_y))
        throw std::runtime_error("unexpected view on y dimension on output buffer");
    if (output_buffer->parent->lengths.t[NN_DATA_COORD_n] != output_buffer->get_length(NN_DATA_COORD_n))
        throw std::runtime_error("unexpected view on n dimension on output buffer");
    if (output_buffer->parent->lengths.t[NN_DATA_COORD_p] != output_buffer->get_length(NN_DATA_COORD_p))
        throw std::runtime_error("unexpected view on p dimension on output buffer");
    if (output_buffer->parent->lengths.t[NN_DATA_COORD_q] != output_buffer->get_length(NN_DATA_COORD_q))
        throw std::runtime_error("unexpected view on q dimension on output buffer");

    if (input_buffer->parent->lengths.t[NN_DATA_COORD_n] != input_buffer->get_length(NN_DATA_COORD_n))
        throw std::runtime_error("unexpected view on n dimension on input buffer");
    if (input_buffer->parent->lengths.t[NN_DATA_COORD_p] != input_buffer->get_length(NN_DATA_COORD_p))
        throw std::runtime_error("unexpected view on p dimension on input buffer");
    if (input_buffer->parent->lengths.t[NN_DATA_COORD_q] != input_buffer->get_length(NN_DATA_COORD_q))
        throw std::runtime_error("unexpected view on q dimension on input buffer");

    auto in_dims = InputDimensions(
            make<InputHeight>(input_buffer->get_length(NN_DATA_COORD_y)),
            make<InputWidth>(input_buffer->get_length(NN_DATA_COORD_x)),
            make<InputFeats>(input_buffer->get_length(NN_DATA_COORD_z)));
    in_full_dims = InputDimensions(
            make<InputHeight>(input_buffer->parent->lengths.t[NN_DATA_COORD_y]),
            make<InputWidth>(input_buffer->parent->lengths.t[NN_DATA_COORD_x]),
            make<InputFeats>(input_buffer->parent->lengths.t[NN_DATA_COORD_z]));
    out_full_dims = OutputDimensions(
            make<OutputHeight>(output_buffer->parent->lengths.t[NN_DATA_COORD_y]),
            make<OutputWidth>(output_buffer->parent->lengths.t[NN_DATA_COORD_x]),
            make<OutputFeats>(output_buffer->parent->lengths.t[NN_DATA_COORD_z]));
    stride_info = {kernel_info.stride,
                   {make<Rows>(input_buffer->view_begin.t[NN_DATA_COORD_y]),
                    make<Cols>(input_buffer->view_begin.t[NN_DATA_COORD_x])}};

    if (kernel_info.center.row < 0)
        throw std::runtime_error("kernel center row is invalid");
    if (kernel_info.center.col < 0)
        throw std::runtime_error("kernel center col is invalid");
        
    kernel_info = {
            {make<KernelHeight>(weights_buffer->get_length(NN_DATA_COORD_y)),
             make<KernelWidth>(weights_buffer->get_length(NN_DATA_COORD_x)),
             make<KernelFeats>(weights_buffer->get_length(NN_DATA_COORD_z))},
             make<OutputFeats>(output_buffer->get_length(NN_DATA_COORD_z)),
            {make<Rows>((uint64_t)kernel_info.center.row), make<Cols>((uint64_t)kernel_info.center.col)},
            stride_info.stride};
    
    prepared_for = std::make_tuple(
        (float*)input_buffer->parent->data_buffer,
        (float*)output_buffer->parent->data_buffer,
        (float*)weights_buffer->parent->data_buffer,
        (float*)bias_buffer->parent->data_buffer);

    compiled_convolution = compileConvolution(
        device->thread_pool.get_num_threads(),
        kernel_info,
        in_dims,
        in_full_dims,
        out_dims,
        out_full_dims,
        stride_info,
        make<InputFeatsStart>(input_buffer->view_begin.t[NN_DATA_COORD_z]),
        make<OutputFeatsStart>(output_buffer->view_begin.t[NN_DATA_COORD_z]),
        output_buffer->get_length(NN_DATA_COORD_n),
        std::get<0>(prepared_for),
        std::get<1>(prepared_for),
        std::get<2>(prepared_for),
        std::get<3>(prepared_for),
        (activation.function == NN_ACTIVATION_FUNCTION_RELU));
}

convolution_f32_batch24n::convolution_f32_batch24n(ValueU64<Batch> batch_size,
                                 OutputDimensions out_dims,
                                 KernelInfo kernel_info,
                                 nn_argument_activation_t activation,
                                 nn_device_internal *device)
    : batch_size(batch_size)
    , out_dims(out_dims)
    , kernel_info(kernel_info)
    , activation(activation)
    , device(device)
    , prepared_for(std::make_tuple(nullptr, nullptr, nullptr, nullptr))
    , in_full_dims(make<InputHeight>(0u), make<InputWidth>(0u), make<InputFeats>(0u))
    , out_full_dims(make<OutputHeight>(0u), make<OutputWidth>(0u), make<OutputFeats>(0u))
    , stride_info(StrideInfo{{make<Rows>(0u), make<Cols>(0u)},
                             { make<Rows>(0u), make<Cols>(0u) } })
    , compiled_convolution(nullptr)
{}

convolution_f32_batch24n::~convolution_f32_batch24n()
{
    if (compiled_convolution)
        release(compiled_convolution);
}

void convolution_f32_batch24n::forward(
    const std::vector<const nn_workload_data_t *> &inputs,
    const std::vector<const nn_workload_data_t *> &parameters,
    const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(parameters.size() == 2);
    assert(outputs.size() == 1);

    forward(nn::workload_data_cast<>(inputs[0]),
            nn::workload_data_cast<>(parameters[0]),
            nn::workload_data_cast<>(parameters[1]),
            nn::workload_data_cast<>(outputs[0]));
}

void convolution_f32_batch24n::prepare_forward(
    const std::vector<const nn_workload_data_t *> &inputs,
    const std::vector<const nn_workload_data_t *> &parameters,
    const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(parameters.size() == 2);
    assert(outputs.size() == 1);

    prepare_forward(nn::workload_data_cast<>(inputs[0]),
                    nn::workload_data_cast<>(parameters[0]),
                    nn::workload_data_cast<>(parameters[1]),
                    nn::workload_data_cast<>(outputs[0]));
}

bool convolution_f32_batch24n::validate_input(size_t index, nn_workload_data_t *data)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> convolution_f32_batch24n::create_inputs(bool allocate_delta)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> convolution_f32_batch24n::create_parameters(bool allocate_delta)
{
    return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIOXY, nn::layout_oblockioxy_f32>::create(
                kernel_info.dims.width,
                kernel_info.dims.height,
                kernel_info.dims.feats,
                out_dims.feats,
                convolution::OUTPUT_ACCEPTED_MOD,
                allocate_delta),
            nn::data_helper<NN_WORKLOAD_DATA_TAG_O, nn::layout_o_f32>::create(device, out_dims.feats, allocate_delta)};
}

std::vector<nn_workload_data_t *> convolution_f32_batch24n::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
                batch_size,
                out_dims.width,
                out_dims.height,
                out_dims.feats,
                BATCH_ACCEPTED_BLOCK,
                allocate_delta)};
}

} // namespace layer

