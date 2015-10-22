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
#include "cpu/core/layer_convolution_avx2_forward.h"
#include "cpu/test_helpers.h"
#include <functional>

namespace
{
using namespace convolution::forward;
using namespace convolution;

uint64_t ceil_div(uint64_t arg, uint64_t div)
{
    return (arg / div) + ((arg % div == 0) ? 0u : 1u);
}

typedef std::function<decltype(ult_nn_naive_convolve)> ConvFunc;
struct BigConvTest : ::testing::TestWithParam<ConvFunc>
{
    std::vector<float> input;
    std::vector<float> weights;
    std::vector<float> output;
    std::vector<float> outputRef;
    uint64_t in_width;
    uint64_t in_height;
    uint64_t out_width;
    uint64_t out_height;
    uint64_t wwidth;
    uint64_t wheight;
    uint64_t in_features;
    uint64_t out_features;

    void prepareAndCalc(ConvFunc conv,
                        uint64_t pwidth,
                        uint64_t pheight,
                        uint64_t pwwidth,
                        uint64_t pwheight,
                        uint64_t input_features,
                        uint64_t output_features,
                        uint64_t col_stride,
                        uint64_t row_stride)
    {
        in_width = pwidth;
        in_height = pheight;
        out_width = ceil_div(in_width, col_stride);
        out_height = ceil_div(in_height, row_stride);
        wwidth = pwwidth;
        wheight = pwheight;
        in_features = input_features;
        out_features = output_features;

        input.resize(in_width * in_height * in_features);
        weights.resize(wwidth * wheight * in_features * out_features);
        output.resize(out_features * out_width * out_height);
        randomizeWithAnyThreaded(input.begin(), input.end(), 8);
        randomizeWithAnyThreaded(weights.begin(), weights.end(), 8);
        std::vector<float> empty_bias(out_features, 0);

        conv(
            &input.front(), &output.front(), &weights.front(), &empty_bias.front(),
            out_features, in_features, out_width, out_height, in_width, in_height, wwidth, wheight,
            col_stride, row_stride, in_width / 2, in_height / 2);
    }

    void compareOutputs()
    {
        for (uint64_t feat = 0u; feat < out_features; ++feat)
            for (uint64_t row = 0u; row < out_height; ++row)
                for (uint64_t col = 0u; col < out_width; ++col)
                    ASSERT_FLOAT_CHECK(outputRef[(feat * out_height + row) * out_width + col],
                                       output[(feat * out_height + row) * out_width + col])
                        << "feat: " << feat << ", row: " << row << ", col: " << col;
    }

    void testFull(uint64_t pwidth,
                  uint64_t pheight,
                  uint64_t pwwidth,
                  uint64_t pwheight,
                  uint64_t input_features,
                  uint64_t output_features,
                  uint64_t col_stride,
                  uint64_t row_stride)
    {
        prepareAndCalc(ult_nn_naive_convolve,
            pwidth, pheight, pwwidth, pwheight, input_features, output_features, col_stride, row_stride);
        outputRef = output;
        memset(&output.front(), 0, sizeof(float) * output.size());
        std::vector<float> empty_bias(output_features, 0);
        ult_nn_convolve_fst_simplified(
            &input.front(), &output.front(), &weights.front(), &empty_bias.front(),
            out_features, in_features, in_width, in_height, wwidth, wheight,
            col_stride, row_stride, in_width / 2, in_height / 2);
        ASSERT_NO_FATAL_FAILURE(compareOutputs());
    }

    void testDifferentParamsFull(std::vector<uint64_t> widths,
                                 std::vector<uint64_t> heights,
                                 std::vector<uint64_t> wwidths,
                                 std::vector<uint64_t> wheights,
                                 std::vector<uint64_t> in_feats,
                                 std::vector<uint64_t> out_feats,
                                 std::vector<uint64_t> col_strides = {1},
                                 std::vector<uint64_t> row_strides = {1})
    {
        using namespace std::placeholders;
        for (uint64_t width : widths) for (uint64_t height : heights)
            for (uint64_t wwidth : wwidths) for (uint64_t wheight : wheights)
                for (uint64_t ifeats : in_feats) for (uint64_t ofeats : out_feats)
                    for (uint64_t col_stride : col_strides) for (uint64_t row_stride : row_strides)
                        ASSERT_NO_FATAL_FAILURE(testFull(
                                width, height, wwidth, wheight,
                                ifeats, ofeats,
                                col_stride, row_stride))
                            << "width: " << width << ", height: " << height
                            << ", weights width: " << wwidth << ", weights height:" << wheight 
                            << ", input features: " << ifeats
                            << ", output features: " << ofeats
                            << ", col stride: " << col_stride 
                            << ", row stride: " << row_stride;
    }
};

TEST_F(BigConvTest, vsNaive_smallParams)
{ 
    testDifferentParamsFull({1, 2, 3, 4, 6},
                            {1, 2, 3, 4, 6},
                            {1, 2, 3, 4, 5},
                            {1, 2, 3, 4, 5},
                            {1, 2, 3, 4, 5},
                            {1, 2, 3, 4, 5, 6});
}
TEST_F(BigConvTest, vsNaive_inputFeatMaps_modExpected)
{ 
    testDifferentParamsFull({4, 5},
                            {4, 5},
                            {2, 3},
                            {2, 3},
                            {INPUT_ACCEPTED_MOD, 2 * INPUT_ACCEPTED_MOD, 3 * INPUT_ACCEPTED_MOD, 4 * INPUT_ACCEPTED_MOD},
                            {OUTPUT_ACCEPTED_MOD});
}
TEST_F(BigConvTest, vsNaive_inputFeatMaps_aroundMod)
{ 
    testDifferentParamsFull({4, 5},
                            {4, 5},
                            {2, 3},
                            {2, 3},
                            {INPUT_ACCEPTED_MOD - 1, INPUT_ACCEPTED_MOD + 1, INPUT_ACCEPTED_MOD + INPUT_ACCEPTED_MOD / 2,
                             2 * INPUT_ACCEPTED_MOD - 1, 2 * INPUT_ACCEPTED_MOD + 1, 2 * INPUT_ACCEPTED_MOD + INPUT_ACCEPTED_MOD / 2,
                             3 * INPUT_ACCEPTED_MOD - 1, 4 * INPUT_ACCEPTED_MOD + 1},
                            {OUTPUT_ACCEPTED_MOD});
}
TEST_F(BigConvTest, vsNaive_outputFeatMaps_modExpected)
{ 
    testDifferentParamsFull({4, 5},
                            {4, 5},
                            {2, 3},
                            {2, 3},
                            {INPUT_ACCEPTED_MOD},
                            {OUTPUT_ACCEPTED_MOD, OUTPUT_ACCEPTED_MOD * 2, OUTPUT_ACCEPTED_MOD * 3, OUTPUT_ACCEPTED_MOD * 4});
}
TEST_F(BigConvTest, vsNaive_outputFeatMaps_aroundMod)
{ 
    testDifferentParamsFull({4, 5},
                            {4, 5},
                            {2, 3},
                            {2, 3},
                            {INPUT_ACCEPTED_MOD},
                            {OUTPUT_ACCEPTED_MOD * 2 - 1, OUTPUT_ACCEPTED_MOD * 2 + 1, OUTPUT_ACCEPTED_MOD * 2 + OUTPUT_ACCEPTED_MOD / 2,
                             3 * OUTPUT_ACCEPTED_MOD - 1, 4 * OUTPUT_ACCEPTED_MOD + 1});
}
TEST_F(BigConvTest, vsNaive_bigFeatMods)
{ 
    testDifferentParamsFull({8},
                            {8},
                            {3},
                            {3},
                            {INPUT_ACCEPTED_MOD * 16},
                            {OUTPUT_ACCEPTED_MOD * 24});
}
TEST_F(BigConvTest, vsNaive_bigXY)
{ 
    testDifferentParamsFull({61},
                            {69},
                            {11},
                            {11},
                            {INPUT_ACCEPTED_MOD},
                            {OUTPUT_ACCEPTED_MOD});
}
TEST_F(BigConvTest, vsNaive_withStride)
{ 
    testDifferentParamsFull({1, 4, 5, 6},
                            {1, 4, 5, 6},
                            {1, 4, 5},
                            {1, 4, 5},
                            {INPUT_ACCEPTED_MOD},
                            {OUTPUT_ACCEPTED_MOD},
                            {2, 3, 4, 5, 6},
                            {2, 3, 4, 5, 6});
}
TEST_F(BigConvTest, xbyak_version)
{ 
    testDifferentParamsFull({1},
                            {1},
                            {1},
                            {1},
                            {8},
                            {4},
                            {1},
                            {1});
}
} //namespace


