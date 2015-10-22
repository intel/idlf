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
#include "forward_convolve_simmplifying_wrapper.h"
#include "naive_implementations.h"
#include "cpu/core/layer_convolution_avx2_forward.h"
#include "common/nn_allocate.h"
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace
{
using namespace convolution::forward;
using namespace convolution;

const uint64_t BUFFER_ALIGNMENT = 4096u;
const uint64_t ACC_REGISTERS = 12u;
const uint64_t BATCH_SHIFT = BUFFERS_ALIGNMENT;
const uint64_t BATCH_BLOCKS = ACC_REGISTERS / OUTPUT_ACCEPTED_MOD;
const uint64_t BATCH = BATCH_SHIFT / sizeof(float) *  BATCH_BLOCKS;
const uint64_t BATCH_SIZE = BATCH * sizeof(float);
static_assert(BATCH == BATCH_ACCEPTED_BLOCK, "invalid batch sizes");


uint64_t ceil_div(uint64_t arg, uint64_t div)
{
    return (arg / div) + ((arg % div == 0) ? 0u : 1u);
}

auto copyToBatched(float* buffer, uint64_t size)
    -> decltype(nn_make_unique_aligned<float>(0))
{
    auto allocated = nn_make_unique_aligned<float>(size * BATCH);
    for (uint64_t i = 0u; i < size; ++i)
        for (uint64_t j = 0u; j < BATCH; ++j)
            allocated.get()[i * BATCH + j] = buffer[i];
    return std::move(allocated);
}

void transposeKernel(
    float* src,
    float* dest,
    KernelDimensions kern_dims, 
    ValueU64<InputFeats> num_input_features, 
    ValueU64<OutputFeats> num_output_features,
    uint64_t size_of_output_features_block)
{
    uint64_t ofeats_blocks = ceil_div(num_output_features, size_of_output_features_block);
    for (uint64_t outfeat = 0u; outfeat < num_output_features; ++outfeat)
    {
        uint64_t ofeat_block = outfeat / size_of_output_features_block;
        uint64_t ofeat_in_block = outfeat % size_of_output_features_block;

        for (uint64_t infeat = 0u; infeat < num_input_features; ++infeat)
            for (uint64_t r = 0u; r < kern_dims.height; ++r)
                for (uint64_t c = 0u; c < kern_dims.width; ++c)
                    dest[(((r * kern_dims.width + c) * ofeats_blocks + ofeat_block)
                            * num_input_features + infeat)
                            * size_of_output_features_block + ofeat_in_block] =
                        src[((outfeat * kern_dims.height + r) * kern_dims.width + c) * num_input_features + infeat];
    }
}

void extractOutputFromBatched(
    float* dest,
    float* src,
    OutputDimensions dims,
    uint64_t batchNum)
{
    for (uint64_t row = 0u; row < dims.height; ++row)
        for (uint64_t col = 0u; col < dims.width; ++col)
            for (uint64_t feat = 0u; feat < dims.feats; ++feat)
                dest[(feat * dims.height + row) * dims.width + col] =
                    src[((row * dims.width + col) * dims.feats + feat) * BATCH + batchNum];
}

void runConvolveBatched(
    float* input_ref,
    float* output_ref,
    float* kernel_ref,
    float* bias_ref,
    OutputDimensions out_dims,
    InputDimensions in_dims,
    KernelInfo kern_info)
{
    auto alignedInput = copyToBatched(input_ref, in_dims.size());
    auto output_blocks = ceil_div(out_dims.feats, OUTPUT_ACCEPTED_MOD);
    auto alignedKernel = nn_make_unique_aligned<float>(
        kern_info.dims.size() * output_blocks * OUTPUT_ACCEPTED_MOD * in_dims.feats);

    transposeKernel(kernel_ref, alignedKernel.get(),
        kern_info.dims, in_dims.feats, out_dims.feats, OUTPUT_ACCEPTED_MOD);


    uint64_t outputSize = out_dims.size() * BATCH;
    auto alignedOutput = nn_make_unique_aligned<float>(outputSize);
    memset(alignedOutput.get(), 0, outputSize * sizeof(float));

    StrideInfo stride_info = {kern_info.stride, {make<Rows>(0), make<Cols>(0)}};

    auto func = compileConvolution(
        4,
        kern_info,
        in_dims,
        in_dims,
        out_dims,
        out_dims,
        stride_info,
        make<InputFeatsStart>(0u),
        make<OutputFeatsStart>(0u),
        1u,
        alignedInput.get(),
        alignedOutput.get(),
        alignedKernel.get(),
        bias_ref);

    convolve_fst_single_threaded(func,
        alignedInput.get(), alignedOutput.get(), alignedKernel.get(), bias_ref,
        make<Batch>(BATCH), kern_info, in_dims, out_dims,
        stride_info,
        make<InputFeatsStart>(0), make<OutputFeatsStart>(0));

    extractOutputFromBatched(output_ref, alignedOutput.get(), out_dims, 11);
    release(func);
    return;
}
} //namespace

void ult_nn_convolve_fst_simplified(
    float* input_ref,
    float* output_ref,
    float* kernel_ref,
    float* bias_ref,
    uint64_t num_output_feature_maps,
    uint64_t num_input_feature_maps,
    uint64_t width,
    uint64_t height,
    uint64_t kernel_width,
    uint64_t kernel_height,
    uint64_t stride_col,
    uint64_t stride_row,
    uint64_t offset_x,
    uint64_t offset_y)
{
    runConvolveBatched(
        input_ref, output_ref, kernel_ref, bias_ref,
        OutputDimensions(
            make<OutputHeight>(ceil_div(height, stride_row)),
            make<OutputWidth>(ceil_div(width, stride_col)),
            make<OutputFeats>(num_output_feature_maps)),
        InputDimensions(make<InputHeight>(height),
                        make<InputWidth>(width),
                        make<InputFeats>(num_input_feature_maps)),
        KernelInfo{KernelDimensions(make<KernelHeight>(kernel_height),
                                    make<KernelWidth>(kernel_width),
                                    make<KernelFeats>(num_input_feature_maps)),
                   make<OutputFeats>(num_output_feature_maps),
                   KernelCenter{make<Rows>(offset_y), make<Cols>(offset_x)},
                   Stride{make<Rows>(stride_row), make<Cols>(stride_col)}});
}

void ult_nn_convolve_fst_simplified_no_offset(
    float* input_ref,
    float* output_ref,
    float* kernel_ref,
    uint64_t num_output_feature_maps,
    uint64_t num_input_feature_maps,
    uint64_t width,
    uint64_t height,
    uint64_t kernel_width,
    uint64_t kernel_height,
    uint64_t stride_col,
    uint64_t stride_row)
{
    std::vector<float> empty_bias(num_output_feature_maps, 0);
    runConvolveBatched(
        input_ref, output_ref, kernel_ref, &empty_bias.front(),
        OutputDimensions(
            make<OutputHeight>(ceil_div(height, stride_row)),
            make<OutputWidth>(ceil_div(width, stride_col)),
            make<OutputFeats>(num_output_feature_maps)),
        InputDimensions(make<InputHeight>(height),
                        make<InputWidth>(width),
                        make<InputFeats>(num_input_feature_maps)),
        KernelInfo{KernelDimensions(make<KernelHeight>(kernel_height),
                                    make<KernelWidth>(kernel_width),
                                    make<KernelFeats>(num_input_feature_maps)),
                   make<OutputFeats>(num_output_feature_maps),
                   KernelCenter{make<Rows>(kernel_height / 2), make<Cols>(kernel_width / 2)},
                   Stride{make<Rows>(stride_row), make<Cols>(stride_col)}});
}

