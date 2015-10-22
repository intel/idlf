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

#include "device/cpu/core/layer_convolution_avx2_batch24n.h"
#include "device/cpu/core/layer_convolution_avx2_forward.h"
#include "device/cpu/api_internal/data_helper.h"
#include "tester/g_ult/unit_tests/cpu/naive_implementations.h"
#include "cpu/test_helpers.h"
#include <cfloat>
#include <iostream>
#include <gtest/gtest.h>

const uint32_t C_simd_width = sizeof(__m256)/sizeof(float);
const uint32_t C_slice_size = 2 * C_simd_width;

namespace
{
using namespace convolution;

struct TestStride : Stride {
    TestStride(Value<Rows, uint64_t> rows = make<Rows>(0),
               Value<Cols, uint64_t> cols = make<Cols>(0))
        : Stride({rows, cols})
    {}
};

struct DataPair
{
    std::vector<float>                          naive;
    std::unique_ptr<nn::workload_data<nn::layout_f32>>   impl;
};

std::shared_ptr<DataPair> make_output(uint32_t batch, uint32_t width, uint32_t height, uint32_t feats)
{
    auto ret = std::make_shared<DataPair>();
    ret->impl = decltype(ret->impl)(
        nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_f32>::create(
            (batch + BATCH_ACCEPTED_BLOCK - 1) / BATCH_ACCEPTED_BLOCK * BATCH_ACCEPTED_BLOCK,
            width, height, feats, BATCH_ACCEPTED_BLOCK));
    memset(ret->impl->parent->data_buffer, 0, sizeof(float) * ret->impl->parent->lengths.size());
    ret->naive = decltype(ret->naive)(batch * width * height * feats, 0);
    return ret;
}

std::shared_ptr<DataPair> make_input(uint32_t batch, uint32_t width, uint32_t height, uint32_t feats)
{
    auto batch_block_size = BATCH_ACCEPTED_BLOCK;

    auto ret = make_output(batch, width, height, feats);

    std::vector<float> aux(ret->naive.size());
    randomizeWithAny(aux.begin(), aux.end(), -10.0, 10.0, batch * width * height * feats);
    auto it = aux.begin();
    for (uint32_t b = 0; b < batch; ++b)
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                for (uint32_t f = 0; f < feats; ++f)
                {
                    ult_nn_naive_convolve_set_input_value(
                        &ret->naive.front(), width, height, feats, b, x, y, f, *it);
                    auto batch_block = b / batch_block_size;
                    auto pic_in_block = b % batch_block_size;
                    ret->impl->at(batch_block, x, y, f, pic_in_block, 0) = *it;
                    ++it;
                }
    return ret;
}

std::shared_ptr<DataPair> make_bias(uint32_t feats)
{
    auto ret = std::make_shared<DataPair>();
    ret->impl = decltype(ret->impl)(
        nn::data_helper<NN_WORKLOAD_DATA_TAG_O, nn::layout_f32>::create(nullptr, feats, false));
    ret->naive = decltype(ret->naive)(feats, 0);

    randomizeWithAny(ret->naive.begin(), ret->naive.end(), -10.0, 10.0, feats);
    for (uint32_t x = 0; x < feats; ++x)
        ret->impl->at(0, x, 0, 0, 0, 0) = ret->naive[x];
    return ret;
}

std::shared_ptr<DataPair> make_kernel(uint32_t width, uint32_t height, uint32_t in_feats, uint32_t out_feats)
{
    auto outfeats_block_size = OUTPUT_ACCEPTED_MOD;

    auto ret = std::make_shared<DataPair>();
    ret->impl = decltype(ret->impl)(
        nn::data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIOXY, nn::layout_f32>::create(
            width, height, in_feats, out_feats, OUTPUT_ACCEPTED_MOD));
    ret->naive = decltype(ret->naive)(width * height * in_feats * out_feats, 0);

    std::vector<float> aux(ret->naive.size());
    randomizeWithAny(aux.begin(), aux.end(), -10.0, 10.0, width * height * in_feats * out_feats);
    auto it = aux.begin();
    for (uint32_t p = 0; p < out_feats; ++p)
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                for (uint32_t z = 0; z < in_feats; ++z)
                {
                    ult_nn_naive_convolve_set_kernel_value(
                        &ret->naive.front(), width, height, in_feats, x, y, z, p, *it);
                    ret->impl->at(0, x, y, z, p % outfeats_block_size, p / outfeats_block_size) = *it;
                    ++it;
                }

    return ret;
}

struct InitializedConv
{
    nn_device_internal internal_device;
    std::shared_ptr<layer::convolution_f32_batch24n> primitive;

    InitializedConv(const nn_workload_data_coords_t& kernel_coords,
                    const uint_least32_t& calculated_in_feats,
                    const nn_workload_data_coords_t& output_coords,
                    Stride stride,
                    uint_least32_t batch_size,
                    NN_ACTIVATION_FUNCTION activation)
        : internal_device()
    {
        uint32_t center_offset_x = (kernel_coords.t[NN_DATA_COORD_x]) / 2;
        uint32_t center_offset_y = (kernel_coords.t[NN_DATA_COORD_y]) / 2;

        primitive = decltype(primitive)(new layer::convolution_f32_batch24n(
            make<Batch>(batch_size),
            OutputDimensions{make<OutputHeight>(output_coords.t[NN_DATA_COORD_y]),
                             make<OutputWidth>(output_coords.t[NN_DATA_COORD_x]),
                             make<OutputFeats>(output_coords.t[NN_DATA_COORD_z])},
            KernelInfo{KernelDimensions{make<KernelHeight>(kernel_coords.t[NN_DATA_COORD_y]),
                                        make<KernelWidth>(kernel_coords.t[NN_DATA_COORD_x]),
                                        make<KernelFeats>(calculated_in_feats)},
                       make<OutputFeats>(output_coords.t[NN_DATA_COORD_z]),
                       KernelCenter{make<Rows>(center_offset_y), make<Cols>(center_offset_x)},
                       stride},
            {activation},
            &internal_device));
    }
};

void ult_nn_convolution_check_outputs(
    nn_workload_data_t* output,
    float* output_ref,
    uint_least32_t feats,
    uint_least32_t width,
    uint_least32_t height,
    uint_least32_t batch_size)
{
    for (uint_least32_t pic = 0; pic < batch_size; pic++)
    {
        for (uint_least32_t outmapa = 0; outmapa < feats; outmapa++)
        {
            for (uint_least32_t row = 0; row < height; row++)
            {
                for (uint_least32_t column = 0; column < width; column++)
                {
                    float ref_value = ult_nn_naive_convolve_get_output_value(
                        output_ref, width, height, feats, pic, column, row, outmapa);

                    auto batch_block = pic / BATCH_ACCEPTED_BLOCK;
                    auto pic_in_block = pic % BATCH_ACCEPTED_BLOCK;
                    float value = nn_workload_data_get<float>(
                        output, batch_block, column, row, outmapa, pic_in_block, 0);
                    ASSERT_FLOAT_CHECK(ref_value, value)
                            << " pic: " << pic
                            << " outmapa: " << outmapa
                            << " row: " << row
                            << " column: " << column;
                }
            }
        }
    }
}

void ult_perform_test(
    uint_least32_t batch_size,
    uint_least32_t out_feats,
    uint_least32_t in_feats,
    uint_least32_t input_width,
    uint_least32_t input_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    Stride stride,
    NN_ACTIVATION_FUNCTION activation )
{
    uint32_t out_width = (input_width + stride.cols - 1) / stride.cols;
    uint32_t out_height = (input_height + stride.rows - 1) / stride.rows;
    auto input = make_input(batch_size, input_width, input_height, in_feats);
    auto output = make_output(batch_size, out_width, out_height, out_feats);
    auto kernel = make_kernel(kernel_width, kernel_height, in_feats, out_feats);
    auto bias = make_bias(out_feats);

    ult_nn_naive_convolve_all_pics(
        &input->naive.front(), &output->naive.front(), &kernel->naive.front(), &bias->naive.front(),
        batch_size,
        out_feats,
        in_feats,
        out_width, out_height,
        input_width, input_height,
        kernel_width, kernel_height,
        stride.cols, stride.rows,
        kernel_width / 2, kernel_height / 2);
    if (activation == NN_ACTIVATION_FUNCTION_RELU)
        for (auto& val: output->naive)
            val = std::max(0.0f, val);

    InitializedConv conv(kernel->impl->parent->lengths,
                         in_feats,
                         output->impl->parent->lengths,
                         stride,
                         (batch_size + BATCH_ACCEPTED_BLOCK - 1) / BATCH_ACCEPTED_BLOCK * BATCH_ACCEPTED_BLOCK,
                         activation);


    conv.primitive->prepare_forward({input->impl.get()}, {kernel->impl.get(), bias->impl.get()}, {output->impl.get()});
    conv.primitive->forward({input->impl.get()}, {kernel->impl.get(), bias->impl.get()}, {output->impl.get()});

    ASSERT_NO_FATAL_FAILURE(ult_nn_convolution_check_outputs(
            output->impl.get(), &output->naive.front(),
            out_feats, out_width, out_height,
            batch_size))
        << " batch: " << batch_size << ", "
        << " out_feats: " << out_feats << ", "
        << " in_feats: " << in_feats << ", "
        << " in_width: " << input_width << ", "
        << " in_height: " << input_height << ", "
        << " kernel_width: " << kernel_width << ", "
        << " kernel_height: " << kernel_height << ", "
        << " col_stride: " << stride.cols << ", "
        << " row_stride: " << stride.rows << ", "
        << " activation: " << activation;
        
}

struct Padding
{
    uint_least32_t top;
    uint_least32_t bottom;
    uint_least32_t left;
    uint_least32_t right;
    uint_least32_t in_depth_start;
    uint_least32_t in_depth_end;
    uint_least32_t out_depth_start;
    uint_least32_t out_depth_end;
};

void ult_perform_padding_test(
    uint_least32_t out_size,
    uint_least32_t kernel_size,
    Stride stride,
    Padding padding)
{
    uint_least32_t truncated_out_feats = 19;
    uint_least32_t out_feats = truncated_out_feats + padding.out_depth_start + padding.out_depth_end;
    uint_least32_t truncated_in_feats = 4;
    uint_least32_t in_feats = padding.in_depth_start + padding.in_depth_end + truncated_in_feats;
    uint_least32_t batch_size = 24u;
    uint_least32_t input_height = padding.top + padding.bottom + stride.rows * out_size;
    uint_least32_t input_width = padding.left + padding.right + stride.cols * out_size;

    auto input = make_input(batch_size, input_width, input_height, in_feats);
    auto output_naive = make_output(batch_size, input_width, input_height, truncated_out_feats);
    auto output_fast = make_output(batch_size, out_size, out_size, out_feats);
    auto kernel_naive = make_kernel(kernel_size, kernel_size, in_feats, truncated_out_feats);
    auto kernel_fast = make_kernel(kernel_size, kernel_size, truncated_in_feats, truncated_out_feats);
    auto bias = make_bias(truncated_out_feats);

    auto outfeats_block_size = OUTPUT_ACCEPTED_MOD;
    for (auto y = 0u; y < kernel_size; ++y)
        for (auto x = 0u; x < kernel_size; ++x)
            for (auto z = 0u; z < truncated_in_feats; ++z)
                for (auto o = 0u; o < truncated_out_feats; ++o)
                    kernel_fast->impl->at(0, x, y, z, o % outfeats_block_size, o / outfeats_block_size)
                        = kernel_naive->impl->at(0, x, y, z + padding.in_depth_start, o % outfeats_block_size, o / outfeats_block_size);

    for (uint32_t b = 0; b < batch_size; ++b)
        for (uint32_t y = 0; y < input_height; ++y)
            for (uint32_t x = 0; x < input_width; ++x)
                for (uint32_t f = 0; f < in_feats; ++f)
                    if ((f < padding.in_depth_start) or (f > in_feats - padding.in_depth_end - 1))
                        ult_nn_naive_convolve_set_input_value(
                            &input->naive.front(), input_width, input_height, in_feats, b, x, y, f, 0);

    ult_nn_naive_convolve_all_pics(
        &input->naive.front(),
        &output_naive->naive.front(),
        &kernel_naive->naive.front(),
        &bias->naive.front(),
        batch_size,
        truncated_out_feats,
        in_feats,
        input_width, input_height,
        input_width, input_height,
        kernel_size, kernel_size,
        1, 1, kernel_size / 2, kernel_size / 2);

    auto input_view = std::make_shared<nn::workload_data<nn::layout_f32>>(
        *input->impl,
        nn_workload_data_coords_t(0, padding.left, padding.top, padding.in_depth_start, 0, 0),
        nn_workload_data_coords_t(input->impl->get_length(NN_DATA_COORD_n) - 1,
                                  input_width - padding.right - 1,
                                  input_height - padding.bottom - 1,
                                  in_feats - padding.in_depth_end - 1,
                                  input->impl->get_length(NN_DATA_COORD_p) - 1,
                                  input->impl->get_length(NN_DATA_COORD_q) - 1));

    auto output_view = std::make_shared<nn::workload_data<nn::layout_f32>>(
        *output_fast->impl,
        nn_workload_data_coords_t(0, 0, 0, padding.out_depth_start, 0, 0),
        nn_workload_data_coords_t(output_fast->impl->get_length(NN_DATA_COORD_n) - 1,
                                  output_fast->impl->get_length(NN_DATA_COORD_x) - 1,
                                  output_fast->impl->get_length(NN_DATA_COORD_y) - 1,
                                  out_feats - padding.out_depth_end - 1,
                                  output_fast->impl->get_length(NN_DATA_COORD_p) - 1,
                                  output_fast->impl->get_length(NN_DATA_COORD_q) - 1));

    ASSERT_EQ(stride.rows * out_size, input_view->get_length(NN_DATA_COORD_y));
    ASSERT_EQ(stride.cols * out_size, input_view->get_length(NN_DATA_COORD_x));
    ASSERT_EQ(in_feats - padding.in_depth_start - padding.in_depth_end, input_view->get_length(NN_DATA_COORD_z));
    InitializedConv conv(kernel_fast->impl->parent->lengths,
                         in_feats - padding.in_depth_start - padding.in_depth_end,
                         output_view->get_length(),
                         stride,
                         batch_size,
                         NN_ACTIVATION_FUNCTION_NONE);
    conv.primitive->prepare_forward({input_view.get()}, {kernel_fast->impl.get(), bias->impl.get()}, {output_view.get()});
    conv.primitive->forward({input_view.get()}, {kernel_fast->impl.get(), bias->impl.get()}, {output_view.get()});

    for (uint_least32_t pic = 0; pic < batch_size; pic++)
    {
        for (uint_least32_t outmapa = 0; outmapa < truncated_out_feats; outmapa++)
        {
            for (uint_least32_t row = 0; row < out_size; row++)
            {
                for (uint_least32_t column = 0; column < out_size; column++)
                {
                    float ref_value = ult_nn_naive_convolve_get_output_value(
                        &output_naive->naive.front(), input_width, input_height, truncated_out_feats,
                        pic,
                        column * stride.cols + padding.left,
                        row * stride.rows + padding.top,
                        outmapa);

                    auto batch_block = pic / BATCH_ACCEPTED_BLOCK;
                    auto pic_in_block = pic % BATCH_ACCEPTED_BLOCK;
                    float value = nn_workload_data_get<float>(
                        output_view.get(), batch_block, column, row, outmapa, pic_in_block, 0);
                    ASSERT_FLOAT_CHECK(ref_value, value)
                            << " pic: " << pic
                            << " outmapa: " << outmapa
                            << " row: " << row
                            << " column: " << column;
                }
            }
        }
    }
}

struct InputSize { uint_least32_t width; uint_least32_t height; };
struct KernelSize { uint_least32_t width; uint_least32_t height; };

typedef std::tuple<
        uint_least32_t, //out_feats
        uint_least32_t, //in_feats
        uint_least32_t, //batch
        NN_ACTIVATION_FUNCTION,
        InputSize,
        KernelSize,
        TestStride
    > ArtificialConvStrideParams;

struct CpuConvolutionArtificial : ::testing::TestWithParam<ArtificialConvStrideParams>
{
};

TEST(CpuConvolutionArtificial, cpu_convolution_stride_example)
{
    ult_perform_test(1, 8, 1, 3, 3, 2, 2, {make<Rows>(1), make<Cols>(1)}, NN_ACTIVATION_FUNCTION_NONE);
}

TEST_P(CpuConvolutionArtificial, cpu_convolution_stride)
{
    ult_perform_test(
         std::get<2>(GetParam()), //batch
         std::get<0>(GetParam()), //ofeats
         std::get<1>(GetParam()), //infeats
         std::get<4>(GetParam()).width, //input
         std::get<4>(GetParam()).height, //input
         std::get<5>(GetParam()).width, //kernel
         std::get<5>(GetParam()).height, //kernel
         std::get<6>(GetParam()), //stride
         std::get<3>(GetParam()) //activation
         );
}

INSTANTIATE_TEST_CASE_P(CpuConvolutionArtificialInst, CpuConvolutionArtificial,
                        ::testing::Combine(
                            ::testing::Values(8u, 16u), //ofeats
                            ::testing::Values(4u), //ifeats
                            ::testing::Values(1u, 8u), //batch
                            ::testing::Values(NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU),
                            ::testing::Values(InputSize{3, 3},
                                              InputSize{8, 8},
                                              InputSize{16, 16},
                                              InputSize{32, 32}),
                            ::testing::Values(KernelSize{1, 1},
                                              KernelSize{2, 2},
                                              KernelSize{3, 2},
                                              KernelSize{2, 3},
                                              KernelSize{1, 1},
                                              KernelSize{1, 2},
                                              KernelSize{2, 1},
                                              KernelSize{3, 1},
                                              KernelSize{3, 3}),
                            ::testing::Values(TestStride(make<Rows>(1), make<Cols>(1)),
                                              TestStride(make<Rows>(2), make<Cols>(2)),
                                              TestStride(make<Rows>(3), make<Cols>(3)),
                                              TestStride(make<Rows>(4), make<Cols>(4)),
                                              TestStride(make<Rows>(2), make<Cols>(3)),
                                              TestStride(make<Rows>(4), make<Cols>(3)))));

TEST(cpu_convolution_real, cpu_convolution_LeNet5)
{
    ult_perform_test(1, 8, 1, 32, 32, 5, 5,  {make<Rows>(1), make<Cols>(1)}, NN_ACTIVATION_FUNCTION_NONE);
    ult_perform_test(1, 16, 8, 14, 14, 5, 5, {make<Rows>(1), make<Cols>(1)}, NN_ACTIVATION_FUNCTION_NONE);
    ult_perform_test(1, 120, 16, 5, 5, 5, 5, {make<Rows>(1), make<Cols>(1)}, NN_ACTIVATION_FUNCTION_NONE);
    ult_perform_test(8, 8, 1, 32, 32, 5, 5,  {make<Rows>(1), make<Cols>(1)}, NN_ACTIVATION_FUNCTION_NONE);
    ult_perform_test(8, 16, 8, 14, 14, 5, 5, {make<Rows>(1), make<Cols>(1)}, NN_ACTIVATION_FUNCTION_NONE);
    ult_perform_test(8, 120, 16, 5, 5, 5, 5, {make<Rows>(1), make<Cols>(1)}, NN_ACTIVATION_FUNCTION_NONE);
}

typedef std::tuple<uint_least32_t, //out_size
                   uint_least32_t, //kernel_size
                   TestStride,
                   Padding> ConvolutionPaddingParams;
struct CpuConvolutionPadding : ::testing::TestWithParam<ConvolutionPaddingParams>
{
};

TEST_P(CpuConvolutionPadding, cpu_convolution_padding_stride)
{
    ult_perform_padding_test(
        std::get<0>(GetParam()), //out size
        std::get<1>(GetParam()), //kernel size
        std::get<2>(GetParam()), //stride
        std::get<3>(GetParam()));
}

INSTANTIATE_TEST_CASE_P(CpuConvolutionPaddingInst, CpuConvolutionPadding,
                        ::testing::Combine(
                            ::testing::Values(1u, 2u, 3u), //out size
                            ::testing::Values(1u, 2u, 3u), //kernel size
                            ::testing::Values(TestStride(make<Rows>(1), make<Cols>(1)),
                                              TestStride(make<Rows>(3), make<Cols>(3)),
                                              TestStride(make<Rows>(4), make<Cols>(3))),
                            ::testing::Values(Padding{2, 3, 2, 3, 0, 0, 0, 0},
                                              Padding{1, 2, 2, 1, 2, 1, 1, 2},
                                              Padding{0, 0, 3, 3, 0, 3, 0, 0},
                                              Padding{3, 3, 0, 0, 1, 0, 0, 0},
                                              Padding{1, 0, 0, 0, 0, 1, 1, 0},
                                              Padding{0, 1, 0, 0, 2, 2, 0, 1},
                                              Padding{0, 0, 1, 0, 3, 3, 3, 4},
                                              Padding{0, 0, 0, 1, 1, 3, 2, 2},
                                              Padding{1, 0, 2, 0, 3, 1, 2, 1},
                                              Padding{0, 2, 0, 1, 1, 1, 1, 1},
                                              Padding{4, 3, 2, 1, 5, 7, 7, 5})));

} //namespace

