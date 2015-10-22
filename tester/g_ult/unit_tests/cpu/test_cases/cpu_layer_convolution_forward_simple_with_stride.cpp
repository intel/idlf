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
typedef std::function<void(float*, float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)> ConvFunc;
struct SimpleStridedConvTest : ::testing::TestWithParam<ConvFunc>
{
    std::vector<float> input;
    std::vector<float> weights;
    std::vector<float> output;
    unsigned width;
    unsigned height;
    unsigned wwidth;
    unsigned wheight;

    unsigned strided_output_size(unsigned size, unsigned shift)
    {
        if (size == 0) return 0;
        return (size - 1) / shift + 1;
    }

    void prepareAndCalc(ConvFunc convFunc,
                        unsigned pwidth,
                        unsigned pheight,
                        unsigned pwwidth,
                        unsigned pwheight,
                        unsigned xstride,
                        unsigned ystride)
    {
        width = pwidth;
        height = pheight;
        wwidth = pwwidth;
        wheight = pwheight;
        input.resize(width * height);
        output.resize(strided_output_size(width, xstride) * strided_output_size(height, ystride));
        weights.resize(wwidth * wheight);
        randomize<decltype(input)>(input, {0.2f, 0.3f, 0.4f, 3.14f, 2.75f});
        randomize<decltype(weights)>(weights, {0.211f, 0.311f, 0.411f, 3.1411f, 2.7511f});

        convFunc(&input.front(), &output.front(), &weights.front(),
                 width, height, wwidth, wheight,
                 xstride, ystride);
    }
};

TEST_P(SimpleStridedConvTest, single_weight_overstrided)
{ 
    prepareAndCalc(GetParam(), 5u, 7u, 1u, 1u, 5u, 7u);
	ASSERT_EQ(1, output.size());
    ASSERT_FLOAT_CHECK(weights.front() * input[0], output[0]);
}

TEST_P(SimpleStridedConvTest, weights_3x3_overstrided)
{ 
    prepareAndCalc(GetParam(), 2u, 2u, 3u, 3u, 3u, 4u);
        
    ASSERT_EQ(1, output.size());
    ASSERT_FLOAT_CHECK(weights[wwidth + 1] * input[0]
              + weights[wwidth + 2] * input[1]
              + weights[2 * wwidth + 1] * input[width + 0]
              + weights[2 * wwidth + 2] * input[width + 1],
              output[0]);
}

TEST_P(SimpleStridedConvTest, single_weight_stride2y1x)
{ 
    prepareAndCalc(GetParam(), 5u, 7u, 1u, 1u, 2u, 1u);
    auto out_width = 3u;
    ASSERT_EQ(out_width * height, output.size());

    for (auto i = 0u; i < height; ++i)
    {
        ASSERT_FLOAT_CHECK(weights.front() * input[i * width], output[i * out_width + 0]) << i;
        ASSERT_FLOAT_CHECK(weights.front() * input[i * width + 2], output[i * out_width + 1]);
        ASSERT_FLOAT_CHECK(weights.front() * input[i * width + 4], output[i * out_width + 2]);
    }
}

TEST_P(SimpleStridedConvTest, single_weight_stride3y2x)
{ 
    auto y_stride = 2u;
    prepareAndCalc(GetParam(), 5u, 7u, 1u, 1u, 3u, y_stride);
    auto out_width = 2u;
    auto out_height = 4u;
	ASSERT_EQ(out_width * out_height, output.size());

    for (auto i = 0u; i < out_height; ++i)
    {
        ASSERT_FLOAT_CHECK(weights.front() * input[i * y_stride * width + 0], output[i * out_width + 0]);
        ASSERT_FLOAT_CHECK(weights.front() * input[i * y_stride * width + 3], output[i * out_width + 1]);
    }
}

TEST_P(SimpleStridedConvTest, single_weight2x2_stride2y3x)
{ 
    auto y_stride = 3u;
    auto x_stride = 2u;
    prepareAndCalc(GetParam(), 5u, 5u, 2u, 2u, x_stride, y_stride);
    auto out_width = 3u;
    auto out_height = 2u;
	ASSERT_EQ(out_width * out_height, output.size());

    ASSERT_FLOAT_CHECK(weights[wwidth + 1] * input[0 * width + 0 * x_stride],
              output[0 * out_width + 0]);

    ASSERT_FLOAT_CHECK(
              weights[wwidth + 0] * input[0 * width + 1 * x_stride - 1]
              + weights[wwidth + 1] * input[0 * width + 1 * x_stride - 0],
              output[0 * out_width + 1]);

    ASSERT_FLOAT_CHECK(
              weights[wwidth + 0] * input[0 * width + 2 * x_stride - 1]
              + weights[wwidth + 1] * input[0 * width + 2 * x_stride - 0],
              output[0 * out_width + 2]);

    ASSERT_FLOAT_CHECK(weights[0 * wwidth + 1] * input[(1 * y_stride - 1) * width + 0 * x_stride]
              + weights[1 * wwidth + 1] * input[(1 * y_stride - 0) * width + 0 * x_stride],
              output[1 * out_width + 0]);

    ASSERT_FLOAT_CHECK(weights[0 * wwidth + 0] * input[(1 * y_stride - 1) * width + 1 * x_stride - 1]
              + weights[0 * wwidth + 1] * input[(1 * y_stride - 1) * width + 1 * x_stride - 0]
              + weights[1 * wwidth + 0] * input[(1 * y_stride - 0) * width + 1 * x_stride - 1]
              + weights[1 * wwidth + 1] * input[(1 * y_stride - 0) * width + 1 * x_stride - 0],
              output[1 * out_width + 1]);

    ASSERT_FLOAT_CHECK(weights[0 * wwidth + 0] * input[(1 * y_stride - 1) * width + 2 * x_stride - 1]
              + weights[0 * wwidth + 1] * input[(1 * y_stride - 1) * width + 2 * x_stride - 0]
              + weights[1 * wwidth + 0] * input[(1 * y_stride - 0) * width + 2 * x_stride - 1]
              + weights[1 * wwidth + 1] * input[(1 * y_stride - 0) * width + 2 * x_stride - 0],
              output[1 * out_width + 2]);
}

using namespace std::placeholders;

INSTANTIATE_TEST_CASE_P(SimpleStridedConvTestInst,
                        SimpleStridedConvTest,
                        ::testing::Values(
                            ult_nn_simplest_convolve_no_depth,
                            std::bind(ult_nn_convolve_fst_simplified_no_offset, _1, _2, _3, 1, 1, _4, _5, _6, _7, _8, _9),
                            std::bind(ult_nn_naive_convolve_simplified, _1, _2, _3, 1, 1, _4, _5, _6, _7, _8, _9)));
} //namespace

