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
#include "cpu/naive_implementations.h"
#include "cpu/forward_convolve_simmplifying_wrapper.h"
#include "cpu/test_helpers.h"
#include <functional>

namespace
{

typedef std::function<decltype(ult_nn_convolve_fst_simplified)> ConvFunc;
struct SimpleConvAllParamsTest : ::testing::TestWithParam<ConvFunc>
{
    std::vector<float> input;
    std::vector<float> weights;
    std::vector<float> output;
};

void ult_nn_naive_convolve_no_out_dims(
    float* input_ref,
    float* output_ref,
    float* kernel_ref,
    float* bias_ref,
    uint64_t num_output_feature_maps,
    uint64_t num_input_feature_maps,
    uint64_t in_width,
    uint64_t in_height,
    uint64_t kernel_width,
    uint64_t kernel_height,
    uint64_t stride_col,
    uint64_t stride_row,
    uint64_t offset_x,
    uint64_t offset_y)
{
    auto strided_output_size = [](unsigned size, unsigned shift)
        {
            if (size == 0) return 0u;
            return (size - 1) / shift + 1;
        };
    uint64_t out_width = strided_output_size(in_width, stride_col);
    uint64_t out_height = strided_output_size(in_height, stride_row);
    ult_nn_naive_convolve(input_ref, output_ref, kernel_ref, bias_ref,
        num_output_feature_maps, num_input_feature_maps,
        out_width, out_height, in_width, in_height, kernel_width, kernel_height,
        stride_col, stride_row, offset_x, offset_y);
}

TEST_P(SimpleConvAllParamsTest, explicitAllParamsTest)
{
    uint64_t width = 3;
    uint64_t height = 2;
    uint64_t wwidth = 2;
    uint64_t wheight = 2;
    uint64_t ifeats = 2;
    uint64_t ofeats = 2;
    std::vector<float> input = {
        2.2f, 5.0f, 7.0f, 2.0f, 7.3f, 1.7f,
        5.3f, 5.5f, 2.7f, 3.3f, 5.7f, 3.5f};

    std::vector<float> filter = {
        3.7f, 2.2f, 7.2f, 3.0f,
        2.7f, 2.5f, 1.1f, 5.1f,
        2.2f, 7.3f, 3.0f, 7.2f,
        2.5f, 7.2f, 5.1f, 1.1f};

    auto in = [&](uint64_t y, uint64_t x, uint64_t feat){
        return input[(y * width + x) * ifeats + feat]; };
    auto kern = [&](uint64_t ofeat, uint64_t y, uint64_t x, uint64_t ifeat){
        return filter[((ofeat * wheight + y) * wwidth + x) * ifeats + ifeat]; };

    auto value = [&](uint64_t in_y, uint64_t in_x, uint64_t kern_y, uint64_t kern_x, uint64_t kern_out) {
            float ret = 0.0;
            for (uint64_t i = 0u; i < ifeats; ++i)
                ret += in(in_y, in_x, i) * kern(kern_out, kern_y, kern_x, i);
            return ret;
        };

    std::vector<float> refOutput = {
        //first output feature
        //first row
            value(0, 0, 0, 0, 0) + value(0, 1, 0, 1, 0) + value(1, 0, 1, 0, 0) + value(1, 1, 1, 1, 0),
            value(0, 1, 0, 0, 0) + value(0, 2, 0, 1, 0) + value(1, 1, 1, 0, 0) + value(1, 2, 1, 1, 0),
            value(0, 2, 0, 0, 0) +                   0  + value(1, 2, 1, 0, 0) +                   0,
        //second row
            value(1, 0, 0, 0, 0) + value(1, 1, 0, 1, 0) + 0,
            value(1, 1, 0, 0, 0) + value(1, 2, 0, 1, 0) + 0,
            value(1, 2, 0, 0, 0) +                   0  + 0,

        //second output feature
        //first row
            value(0, 0, 0, 0, 1) + value(0, 1, 0, 1, 1) + value(1, 0, 1, 0, 1) + value(1, 1, 1, 1, 1),
            value(0, 1, 0, 0, 1) + value(0, 2, 0, 1, 1) + value(1, 1, 1, 0, 1) + value(1, 2, 1, 1, 1),
            value(0, 2, 0, 0, 1) +                   0  + value(1, 2, 1, 0, 1) +                   0,
        //second row
            value(1, 0, 0, 0, 1) + value(1, 1, 0, 1, 1) + 0,
            value(1, 1, 0, 0, 1) + value(1, 2, 0, 1, 1) + 0,
            value(1, 2, 0, 0, 1) +                   0  + 0
    };

    std::vector<float> output(refOutput.size(), 0);

    std::vector<float> empty_bias(ofeats, 0);
    GetParam()(&input.front(), &output.front(), &filter.front(), &empty_bias.front(),
        ofeats, ifeats, width, height, wwidth, wheight, 1, 1, 0, 0);

    for (auto i = 0u; i < output.size(); ++i)
        EXPECT_FLOAT_CHECK(refOutput[i], output[i]) << " row: " << (i / width) << " col: " << (i % width);
}

using namespace std::placeholders;
INSTANTIATE_TEST_CASE_P(SimpleConvAllParamsTestInst,
                        SimpleConvAllParamsTest,
                        ::testing::Values(ult_nn_convolve_fst_simplified, ult_nn_naive_convolve_no_out_dims));

} //namespace

