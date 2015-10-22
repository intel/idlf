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
typedef std::function<void(float*, float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)> ConvFunc;
struct SimpleWithDepthConvTest : ::testing::TestWithParam<ConvFunc>
{
    std::vector<float> input;
    std::vector<float> weights;
    std::vector<float> output;
    unsigned width;
    unsigned height;
    unsigned depth;
    unsigned wwidth;
    unsigned wheight;

    void prepareAndCalc(ConvFunc convFunc,
                        unsigned pwidth,
                        unsigned pheight,
                        unsigned pdepth,
                        unsigned pwwidth,
                        unsigned pwheight)
    {
        width = pwidth;
        height = pheight;
        depth = pdepth;
        wwidth = pwwidth;
        wheight = pwheight;
        input.resize(width * height * pdepth);
        output.resize(width * height);
        weights.resize(wwidth * wheight * pdepth);
        randomizeWithAny(input.begin(), input.end());
        randomizeWithAny(weights.begin(), weights.end());

        convFunc(&input.front(), &output.front(), &weights.front(),
                 depth, width, height, wwidth, wheight);
    }

    void test_depth_only_weight(ConvFunc convFunc)
    {
        prepareAndCalc(convFunc, 5u, 7u, 2u, 1u, 1u);

        for (auto row = 0u; row < height; ++row)
            for (auto col = 0u; col < width; ++col)
                ASSERT_FLOAT_CHECK(output[row * width + col], conv(0, 0, row, col)) << "(" << row << ", " << col << ")";
    }

    float conv(unsigned w_row, unsigned w_col, unsigned i_row, unsigned i_col)
    {
        float ret = 0.0f;

        for (uint64_t feature = 0u; feature < depth; ++feature)
            ret += weights[w_row * wwidth * depth + w_col * depth + feature]
                    * input[i_row * width * depth + i_col * depth + feature];
        return ret;
    }

    void test_weights_3x3(ConvFunc convFunc)
    {
        prepareAndCalc(convFunc, 2u, 2u, 10u, 3u, 3u);
        
        ASSERT_FLOAT_CHECK(conv(1, 1, 0, 0) + conv(1, 2, 0, 1) + conv(2, 1, 1, 0) + conv(2, 2, 1, 1),
                        output[0]);

        ASSERT_FLOAT_CHECK(conv(1, 0, 0, 0) + conv(1, 1, 0, 1) + conv(2, 0, 1, 0) + conv(2, 1, 1, 1),
                        output[1]);

        ASSERT_FLOAT_CHECK(conv(0, 1, 0, 0) + conv(0, 2, 0, 1) + conv(1, 1, 1, 0) + conv(1, 2, 1, 1),
                        output[2]);

        ASSERT_FLOAT_CHECK(conv(0, 0, 0, 0) + conv(0, 1, 0, 1) + conv(1, 0, 1, 0) + conv(1, 1, 1, 1),
                        output[3]);
    }

    void test_weights_3x3_in_bigger_full_only(ConvFunc convFunc)
    {
        for (unsigned w = 5u; w < 10u; ++w)
            for (unsigned h = 5u; h < 10u; ++h)
                ASSERT_NO_FATAL_FAILURE(test_weights_3x3_in_bigger_full_only_single(convFunc, w, h))
                    << "width: " << w << " height: " << h;
    }
    
    void test_weights_3x3_in_bigger_full_only_single(ConvFunc convFunc, unsigned pwidth, unsigned pheight)
    {
        prepareAndCalc(convFunc, pwidth, pheight, 256u * 4u, 3u, 3u);

        for (auto r = 1u; r < height - 1; ++r)
            for (auto c = 1u; c < width - 1; ++c)
                ASSERT_FLOAT_CHECK(conv(0, 0, r - 1, c - 1) + conv(0, 1, r - 1, c) + conv(0, 2, r - 1, c + 1)
                                + conv(1, 0, r, c - 1) + conv(1, 1, r, c) + conv(1, 2, r, c + 1)
                                + conv(2, 0, r + 1, c - 1) + conv(2, 1, r + 1, c) + conv(2, 2, r + 1, c + 1),
                                output[r * width + c]);
    }

    void test_weights_2x2(ConvFunc convFunc)
    {
        prepareAndCalc(convFunc, 2u, 2u, 20u, 2u, 2u);

        ASSERT_FLOAT_CHECK(conv(1, 1, 0, 0),
                  output[0 * width + 0]);

        ASSERT_FLOAT_CHECK(conv(1, 0, 0, 0) + conv(1, 1, 0, 1),
                  output[0 * width + 1]);

        ASSERT_FLOAT_CHECK(conv(0, 1, 0, 0) + conv(1, 1, 1, 0),
                  output[1 * width + 0]);

        ASSERT_FLOAT_CHECK(conv(0, 0, 0, 0) + conv(0, 1, 0, 1) + conv(1, 0, 1, 0) + conv(1, 1, 1, 1),
                  output[1 * width + 1]);
    }

    void test_weights_2x2_odd_sizes(ConvFunc convFunc)
    {
        prepareAndCalc(convFunc, 3u, 5u, 3u, 2u, 2u);

        ASSERT_FLOAT_CHECK(conv(1, 1, 0, 0),
                  output[0 * width + 0]);

        ASSERT_FLOAT_CHECK(conv(1, 0, 0, 0) + conv(1, 1, 0, 1),
                  output[0 * width + 1]);

        ASSERT_FLOAT_CHECK(conv(0, 1, 0, 0) + conv(1, 1, 1, 0),
                  output[1 * width + 0]);

        for (auto r = 1u; r < height; ++r)
            for (auto c = 1u; c < width; ++c)
                ASSERT_FLOAT_CHECK(conv(0, 0, r - 1u, c - 1u)
                                + conv(0, 1, r - 1, c)
                                + conv(1, 0, r, c - 1)
                                + conv(1, 1, r, c),
                          output[r * width + c]);
    }
};

TEST_P(SimpleWithDepthConvTest, small_depth_single_weight)
{ 
    ASSERT_NO_FATAL_FAILURE(test_depth_only_weight(GetParam()));
}

TEST_P(SimpleWithDepthConvTest, weights_3x3)
{
    ASSERT_NO_FATAL_FAILURE(test_weights_3x3(GetParam()));
}

TEST_P(SimpleWithDepthConvTest, weights_2x2)
{
    ASSERT_NO_FATAL_FAILURE(test_weights_2x2(GetParam()));
}

TEST_P(SimpleWithDepthConvTest, weights_2x2_odd_sizes)
{
    ASSERT_NO_FATAL_FAILURE(test_weights_2x2_odd_sizes(GetParam()));
}

TEST_P(SimpleWithDepthConvTest, weights_3x3_in_bigger)
{
    ASSERT_NO_FATAL_FAILURE(
        test_weights_3x3_in_bigger_full_only(GetParam()));
}

using namespace std::placeholders;
INSTANTIATE_TEST_CASE_P(SimpleWithDepthConvTestInst,
                        SimpleWithDepthConvTest,
                        ::testing::Values(
                            ult_nn_simplest_convolve_no_stride_depth_of_input,
                            std::bind(ult_nn_convolve_fst_simplified_no_offset, _1, _2, _3, 1, _4, _5, _6, _7, _8, 1, 1),
                            std::bind(ult_nn_naive_convolve_simplified, _1, _2, _3, 1, _4, _5, _6, _7, _8, 1, 1)));

} //namespace

