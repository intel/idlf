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
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_fully_connected_avx2_batch24n.h"
#include "layer_convolution_avx2_forward.h"
#include "jit_conv_generic.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include "device/cpu/api_internal/data_helper.h"
#include "device/common/nn_asmjit_compilation.h"

namespace layer
{

namespace
{

const auto BATCH_FC_ACCEPTED_BLOCK = 24u;
const auto BATCH_FC_BLOCKS = 3u;
const auto OUTPUT_FEAT_BLOCK_SIZE = 4u;
const auto ACC_REGISTERS = 12u;// BATCH_FC_BLOCKS * OUTPUT_FEAT_BLOCK_SIZE;
const uint64_t BATCH_SIZE = BATCH_FC_ACCEPTED_BLOCK * sizeof(float);

uint32_t ceil_div(uint32_t arg, uint32_t div)
{
    return (arg + div - 1) / div;
}

typedef std::function<void(asmjit::X86Assembler& a)> CalcFunc;
CalcFunc operator and(CalcFunc left, CalcFunc right)
{
    return [=](asmjit::X86Assembler& a){ left(a); right(a); };
}

void createInternalForOutBlock(asmjit::X86Assembler& a,
                                 asmjit::X86GpReg filterReg,
                                 asmjit::X86GpReg inputReg,
                                 uint64_t numOfIfs,
                                 uint64_t numOfOfs)
{
    using namespace asmjit;
    using namespace asmjit::x86;

    if (BATCH_FC_BLOCKS == 6)
    {
        const auto FREE_REGS = 3;
        if (numOfOfs > 2u)
            throw std::runtime_error("implementation error: batch48 ofeat block is greater than 2 (" + std::to_string(numOfOfs) + ")");
        if (numOfOfs == 0) throw std::runtime_error("implementation error: num of out feats to calc is 0");

        for (uint64_t ifeat = 0u; ifeat < numOfIfs; ++ifeat)
        {
            X86Mem filterPtr1 = ptr(filterReg, (ifeat * OUTPUT_FEAT_BLOCK_SIZE + 0) * sizeof(float));
            X86Mem filterPtr2 = ptr(filterReg, (ifeat * OUTPUT_FEAT_BLOCK_SIZE + 1) * sizeof(float));
            a.vbroadcastss(ymm14, filterPtr1);
            if (numOfOfs > 1) a.vbroadcastss(ymm15, filterPtr2);

            for (uint64_t i = 0u; i < BATCH_FC_BLOCKS / 2; ++i)
            {
                a.vmovaps(ymm12, ptr(inputReg, (BATCH_FC_BLOCKS * ifeat + 2 * i) * BATCH_SHIFT * sizeof(float)));
                a.vmovaps(ymm13, ptr(inputReg, (BATCH_FC_BLOCKS * ifeat + 2 * i + 1) * BATCH_SHIFT * sizeof(float)));

                a.vfmadd231ps(ymm(2 * i), ymm12, ymm14);
                a.vfmadd231ps(ymm(2 * i + 1), ymm13, ymm14);
                if (numOfOfs > 1) a.vfmadd231ps(ymm(2 * i + BATCH_FC_BLOCKS), ymm12, ymm15);
                if (numOfOfs > 1) a.vfmadd231ps(ymm(2 * i + 1 + BATCH_FC_BLOCKS), ymm13, ymm15);
            }
        }
    }
    else if (BATCH_FC_BLOCKS == 3)
    {
        for (uint64_t i = 0u; i < numOfIfs; ++i)
        {
            for (uint64_t j = 0; j < BATCH_FC_BLOCKS; ++j)
                a.vmovaps(ymm(ACC_REGISTERS + j), ptr(inputReg, (BATCH_FC_BLOCKS * i + j) * BATCH_SHIFT * sizeof(float)));
            for (uint64_t j = 0u; j < numOfOfs; ++j)
            {
                X86Mem filterPtr = ptr(filterReg, (j + i * OUTPUT_FEAT_BLOCK_SIZE) * sizeof(float));
                a.vbroadcastss(ymm15, filterPtr);
                for (uint64_t k = 0; k < BATCH_FC_BLOCKS; ++k)
                    a.vfmadd231ps(ymm(BATCH_FC_BLOCKS * j + k), ymm(ACC_REGISTERS + k), ymm15);
            }
        }

    }
    else
        throw std::runtime_error("fc-jit invalid batch size");
}

struct InputData
{
    float* input;
    float* output;
    float* filter;
    float* bias;
};

void compile_impl(asmjit::X86Assembler& a,
             bool apply_relu,
             bool load_bias,
             uint64_t total_input_feats,
             uint64_t input_feats,
             uint64_t output_feats,
             uint64_t input_block_size)
{
    using namespace asmjit;
    using namespace asmjit::x86;
    using namespace std::placeholders;
    using namespace nn::asm_compilation;

    auto accBlockPtr = nn_asmjit_param_ptr(InputData, output);
    auto inputBlockPtr = nn_asmjit_param_ptr(InputData, input);
    auto filterBlockPtr = nn_asmjit_param_ptr(InputData, filter);
    auto biasBlockPtr = nn_asmjit_param_ptr(InputData, bias);

    auto accPtr = r12;
    auto inputPtr = r13;
    auto kernelPtr = r14;
    auto biasPtr = r15;

    auto in_block = r8;
    auto out_block = r9;

    a.mov(inputPtr, inputBlockPtr);
    a.mov(kernelPtr, filterBlockPtr);
    a.mov(biasPtr, biasBlockPtr);

    assert(input_feats > 0u);
    auto infeats_blocks = ceil_div(input_feats, input_block_size);
    auto last_in_feats = input_feats - (infeats_blocks - 1) * input_block_size;
    assert(output_feats > 0u);
    auto outfeats_blocks = ceil_div(output_feats, OUTPUT_FEAT_BLOCK_SIZE);
    auto last_out_feats = output_feats - (outfeats_blocks - 1) * OUTPUT_FEAT_BLOCK_SIZE;

    auto calc_out_block = [&](bool load_bias, bool apply_relu, uint64_t in_feats, uint64_t out_feats){
            if (load_bias)
            {
                for (uint64_t i = 0u; i < out_feats; ++i)
                    for (uint64_t k = 0; k < BATCH_FC_BLOCKS; ++k)
                        a.vbroadcastss(ymm(BATCH_FC_BLOCKS * i + k), ptr(biasPtr, i * sizeof(float)));
            }
            else
            {
                for (uint64_t i = 0u; i < out_feats * BATCH_FC_BLOCKS; ++i)
                    a.vmovaps(ymm(i), ptr(accPtr, i * BATCH_SHIFT * sizeof(float)));
            }
            createInternalForOutBlock(a, kernelPtr, inputPtr, in_feats, out_feats);
            if (apply_relu)
            {
                a.vxorps(ymm15, ymm15, ymm15);
                for (uint64_t i = 0u; i < out_feats * BATCH_FC_BLOCKS; ++i)
                    a.vmaxps(ymm(i), ymm(i), ymm15);
            }
            for (uint64_t i = 0u; i < out_feats * BATCH_FC_BLOCKS; ++i)
                a.vmovaps(ptr(accPtr, i * BATCH_SHIFT * sizeof(float)), ymm(i));
        };
    auto calc_full_out_block = [&](bool load_bias, bool apply_relu, uint64_t in_feats){
            a.mov(accPtr, accBlockPtr);
            a.mov(biasPtr, biasBlockPtr);
            a.mov(kernelPtr, filterBlockPtr);
            a.mov(rax, in_block);
            a.mov(rdx, input_block_size * OUTPUT_FEAT_BLOCK_SIZE * sizeof(float));
            a.mul(rdx);
            a.add(kernelPtr, rax);
            loop(out_block, 0, outfeats_blocks - 1,
                    [&](asmjit::X86Assembler& a){
                        calc_out_block(load_bias, apply_relu, in_feats, OUTPUT_FEAT_BLOCK_SIZE);
                        a.add(accPtr, OUTPUT_FEAT_BLOCK_SIZE * BATCH_FC_ACCEPTED_BLOCK * sizeof(float));
                        a.add(kernelPtr, OUTPUT_FEAT_BLOCK_SIZE * total_input_feats * sizeof(float));
                        a.add(biasPtr, OUTPUT_FEAT_BLOCK_SIZE * sizeof(float));
                    }
                )(a);
            calc_out_block(load_bias, apply_relu, in_feats, last_out_feats);
            a.add(inputPtr, in_feats * BATCH_FC_ACCEPTED_BLOCK * sizeof(float));
        };

    if (input_feats <= input_block_size)
    {
        a.mov(in_block, 0);
        calc_full_out_block(true, true, input_feats);
    }
    else
    {
        a.mov(in_block, 0);
        calc_full_out_block(load_bias, false, input_block_size);
        if (infeats_blocks > 2)
        {
            loop(in_block, 1, infeats_blocks - 1,
                [&](asmjit::X86Assembler& a){ calc_full_out_block(false, false, input_block_size); }
            )(a);
        }
        calc_full_out_block(false, apply_relu, last_in_feats);
    }
}

typedef void (*SingleFullyConnectedFunc)(InputData*);

inline void process_job(SingleFullyConnectedFunc func,
                        uint32_t output_feats,
                        uint32_t input_feats,
                        uint32_t BATCH_FC_BLOCKS,
                        float* input,
                        float* output,
                        float* bias,
                        float* weights)
{
    InputData data = {input, output, weights, bias};
    for (auto b = 0u; b < BATCH_FC_BLOCKS; ++b)
    {
        func(&data);
        data.input += input_feats * BATCH_FC_ACCEPTED_BLOCK;
        data.output += output_feats * BATCH_FC_ACCEPTED_BLOCK;
    }
}

} //namespace

void fully_connected_f32_batch24n::forward(const nn::workload_data<> *input_buffer,
                                           const nn::workload_data<> *weights_buffer,
                                           const nn::workload_data<> *bias_buffer,
                                           nn::workload_data<> *output_buffer)
{
    if (prepared_for != std::make_tuple((float*)input_buffer->parent->data_buffer,
                                        (float*)output_buffer->parent->data_buffer,
                                        (float*)weights_buffer->parent->data_buffer,
                                        (float*)bias_buffer->parent->data_buffer))
        throw std::runtime_error("fully connected forward called on different buffers than prepared for");

    for (auto job : compiled->jobs)
        device->thread_pool.push_job(job);

}

void fully_connected_f32_batch24n::prepare_forward(
    const nn::workload_data<> *input_buffer,
    const nn::workload_data<> *weights_buffer,
    const nn::workload_data<> *bias_buffer,
    nn::workload_data<> *output_buffer)
{
    assert(input_buffer->get_length(NN_DATA_COORD_n) == ceil_div(batch_size, BATCH_FC_ACCEPTED_BLOCK));
    assert(output_buffer->get_length(NN_DATA_COORD_n) == ceil_div(batch_size, BATCH_FC_ACCEPTED_BLOCK));
    assert(input_buffer->get_length(NN_DATA_COORD_p) == BATCH_FC_ACCEPTED_BLOCK);
    assert(output_buffer->get_length(NN_DATA_COORD_p) == BATCH_FC_ACCEPTED_BLOCK);

    uint32_t input_feats = input_buffer->get_length(NN_DATA_COORD_x)
        * input_buffer->get_length(NN_DATA_COORD_y) * input_buffer->get_length(NN_DATA_COORD_z);
    uint32_t output_feats = output_buffer->get_length(NN_DATA_COORD_x)
        * output_buffer->get_length(NN_DATA_COORD_y) * output_buffer->get_length(NN_DATA_COORD_z);

    assert(weights_buffer->get_length(NN_DATA_COORD_x) == input_feats);
    assert(bias_buffer->get_length(NN_DATA_COORD_x) == output_feats);

    if (input_buffer->parent->lengths != input_buffer->get_length())
        throw std::runtime_error("unexpected view on input buffer to fully connected");
    if (output_buffer->parent->lengths != output_buffer->get_length())
        throw std::runtime_error("unexpected curr_output1 + view on output buffer to fully connected");

    auto input = (float*)input_buffer->parent->data_buffer;
    auto output = (float*)output_buffer->parent->data_buffer;
    auto weights = (float*)weights_buffer->parent->data_buffer;
    auto bias = (float*)bias_buffer->parent->data_buffer;

    assert(((uint64_t)input) % BATCH_SHIFT == 0);
    assert(((uint64_t)output) % BATCH_SHIFT == 0);
    assert(((uint64_t)weights) % BATCH_SHIFT == 0);
    assert(((uint64_t)bias) % BATCH_SHIFT == 0);

    const auto apply_relu = (activation.function == NN_ACTIVATION_FUNCTION_RELU);
    const auto num_of_threads = device->thread_pool.get_num_threads();

    auto outfeats_block_size = 4u;
    while (outfeats_block_size < 32u
            and (output_feats % (outfeats_block_size * 2u) == 0))
        outfeats_block_size *= 2;
    compiled.reset(
        new jit_convolution_generic(
            input_buffer->get_length(NN_DATA_COORD_n),
            apply_relu,
            num_of_threads,
            output, output_feats, input, input_feats, weights, bias,
            128u,
            outfeats_block_size));


    prepared_for = std::make_tuple(input, output, weights, bias);
}

fully_connected_f32_batch24n::fully_connected_f32_batch24n(
            size_t num_input,
            size_t num_output,
            const nn_argument_activation_t &activation,
            size_t batch_size,
            nn_device_internal *device)
    : num_input(num_input)
    , num_output(num_output)
    , activation(activation)
    , batch_size(batch_size)
    , device(device)
{
}

void fully_connected_f32_batch24n::forward(
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

void fully_connected_f32_batch24n::prepare_forward(
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

bool fully_connected_f32_batch24n::validate_input(size_t index, nn_workload_data_t *data)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> fully_connected_f32_batch24n::create_inputs(bool allocate_delta)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> fully_connected_f32_batch24n::create_parameters(bool allocate_delta)
{
    return{nn::data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIO, nn::layout_oblockio_f32>::create(
                    device, OUTPUT_FEAT_BLOCK_SIZE, num_input, num_output, allocate_delta),
           nn::data_helper<NN_WORKLOAD_DATA_TAG_O, nn::layout_o_f32>::create(device, num_output, allocate_delta)};
}

std::vector<nn_workload_data_t *> fully_connected_f32_batch24n::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
                batch_size,
                1,
                1,
                num_output,
                BATCH_FC_ACCEPTED_BLOCK,
                allocate_delta)};
}

} // namespace layer

