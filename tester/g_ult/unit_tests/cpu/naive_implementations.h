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

#ifndef naive_implementations
#define naive_implementations

#include <immintrin.h>
#include <cmath>
#include <algorithm>

#include "gtest/gtest.h"

#include "../../../../device/api/nn_device_interface_0.h"
#include "../../../../device/cpu/core/layer_softmax_avx2.h"
#include "../../../../device/api/nn_device_api.h"
#include "../../../../device/cpu/core/fixedpoint/layer_fully_connected_int16_fixedpoint_avx2.h"
#include "../../../../device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "../../../../device/cpu/core/layer_normalization_avx2.h"

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp
#define NUM_MERGED_CONVOLUTIONS 2

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation
//cpu_layer_normalization_response_across_maps_fixedpoint.cpp
//cpu_layer_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_pooling_int16_fixedpoint_avx2_compilation.cpp
//cpu_layer_convolution_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_int16_fixedpoint_avx2_compilation.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2_compilation.cpp
void ult_nn_merge_convolution_fixedpoint_naive_set_input_value(
    int16_t* input_ref,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t input_column,
    uint_least32_t input_row,
    uint_least32_t input_map,
    int16_t value,
    uint_least32_t& offset);

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation
//cpu_layer_convolution_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_int16_fixedpoint_avx2_compilation.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2_compilation.cpp
void ult_nn_merge_convolution_fixedpoint_naive_set_kernel_value(
    int16_t* kernel_ref,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t kernel_column,
    uint_least32_t kernel_row,
    uint_least32_t kernel_input_map,
    uint_least32_t kernel_output_map,
    int16_t value,
    uint_least32_t& offset);

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #3
void ult_nn_merge_convolution_fixedpoint_both_initialize_matrices(
    int16_t* input,
    int16_t* output,
    int32_t* biases,
    int16_t* kernel,
    int16_t* input_ref,
    int16_t* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t center_x,
    uint_least32_t center_y
    );

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #4
bool ult_nn_merge_convolution_fixedpoint_check_outputs(
    nn::data<int16_t, 3>* output,
    int16_t* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t center_x,
    uint_least32_t center_y,
    int8_t merge_axis
    );

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #5
//cpu_layer_convolution_int16_fixedpoint_avx2.cpp
void ult_nn_convolve_fp_naive(
    int16_t* input_ref,
    int16_t* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    uint_least32_t center_x,
    uint_least32_t center_y,
    NN_ACTIVATION_FUNCTION activation);

//cpu_layer_normalization.cpp
bool compare_results(nn_workload_item *&work_item, nn::data<float, 4> &output_ref);


//cpu_layer_normalization.cpp
template <class T_primitive>
bool run_naive_layer_normalization(
    float alpha, float beta, uint32_t k, uint32_t n, const nn::data<float, 4> &input, nn::data<float, 4> &output);

template <>
bool run_naive_layer_normalization<layer::normalization_elementwise_linear_f32>(
    float alpha, float beta, uint32_t k, uint32_t n, const nn::data<float, 4> &input, nn::data<float, 4> &output);

template <>
bool run_naive_layer_normalization<layer::normalization_response_across_maps_f32>(
    float alpha, float beta, uint32_t k, uint32_t n, const nn::data<float, 4> &input, nn::data<float, 4> &output);

//cpu_layer_normalization_response_across_maps_fixedpoint.cpp
void ult_nn_lrn_fp_naive(
    int16_t * input_ref,
    int16_t * output_ref,
    int32_t num_feature_maps,
    int32_t feature_map_width,
    int32_t feature_map_height,
    int32_t batch_size,
    int32_t input_fraction,
    int32_t output_fraction,
    float coeff_alpha,
    float coeff_beta,
    float coeff_k
    );

//cpu_layer_normalization_response_across_maps_fixedpoint_compilation.cpp
// Naive normalization_lrn
void ult_nn_lrn_fp_comp_naive(
    int16_t * input_ref,
    int16_t * output_ref,
    int32_t num_feature_maps,
    int32_t feature_map_width,
    int32_t feature_map_height,
    int32_t batch_size,
    int32_t input_fraction,
    int32_t output_fraction,
    float coeff_alpha,
    float coeff_beta,
    float coeff_k
    );

//cpu_layer_pooling_avx2.cpp
void run_reference(nn::workload_data<float> *input,
    nn::data<float, 4> &output,
    size_t pool_size_x,
    size_t pool_size_y,
    size_t pool_stride_x,
    size_t pool_stride_y);

//cpu_layer_pooling_int16_fixedpoint_avx2.cpp
/*static*/ void ult_nn_maxpooling_naive(
    int16_t* input_ref,
    int16_t* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t output_feature_map_width_int,
    uint_least32_t output_feature_map_height_int,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t pool_width,
    uint_least32_t pool_height,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    uint_least32_t center_x,
    uint_least32_t center_y);

//cpu_layer_softmax.cpp
void cpu_layer_softmax(
    nn_workload_item* &work_item,
    bool is_ref,
    nn_device_t *device);

//cpu_fully_connected_int16_avx2.cpp
/*static*/ void ult_nn_fc_naive(
    int16_t* input,
    int32_t* output,
    int32_t* biases,
    int16_t* kernel,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size,
    NN_ACTIVATION_FUNCTION activation);

//cpu_layer_convolution_avx2 #1
//cpu_layer_convolution_pooling_avx2.cpp #1
float ult_nn_convolution_naive_get_output_value(
    float* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t batch,
    uint_least32_t& offset);

//cpu_layer_convolution_avx2 #2
//cpu_layer_convolution_pooling_avx2.cpp #2
void ult_nn_convolution_naive_set_output_value(
    float* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset);

//cpu_layer_convolution_avx2 #3
//cpu_layer_convolution_pooling_avx2.cpp #3
void ult_nn_convolution_naive_set_input_value(
    float* input_ref,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_column,
    uint_least32_t input_row,
    uint_least32_t input_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset);

//cpu_layer_convolution_avx2 #4
//cpu_layer_convolution_pooling_avx2.cpp #4
void ult_nn_convolution_naive_set_kernel_value(
    float* kernel_ref,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t kernel_column,
    uint_least32_t kernel_row,
    uint_least32_t kernel_input_map,
    uint_least32_t kernel_output_map,
    float value,
    uint_least32_t& offset);

//cpu_layer_convolution_avx2 #5
//cpu_layer_convolution_pooling_avx2.cpp #5
void ult_nn_convolve_naive(
    float* input_ref,
    float* output_ref,
    float* biases_ref,
    float* kernel_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation);

//cpu_layer_convolution_pooling_avx2.cpp #6
void ult_nn_maxpooling_naive_convolution_pooling(
    float* input_ref,
    float* output_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t pool_width,
    uint_least32_t pool_height,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    NN_ACTIVATION_FUNCTION activation);

//cpu_layer_convolution_int16_fixedpoint_avx2.cpp #1
int16_t ult_nn_convolution_fp_naive_get_output_value(
    int16_t* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t& offset);

//cpu_layer_convolution_int16_fixedpoint_avx2.cpp #2
/*static*/ void ult_nn_convolution_fp_naive_set_output_value(
    int16_t* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    int16_t value,
    uint_least32_t& offset);

//cpu_layer_convolution_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2_compilation.cpp
void ult_nn_maxpooling_naive_pooling_int16_fixedpoint(
    int16_t* input_ref,
    int16_t* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t output_feature_map_width_int,
    uint_least32_t output_feature_map_height_int,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    uint_least32_t pool_width,
    uint_least32_t pool_height,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    uint_least32_t center_x,
    uint_least32_t center_y,
    NN_ACTIVATION_FUNCTION activation);

//cpu_layer_fullyconnected.cpp
void cpu_layer_fullyconnected(
    nn_workload_item* &work_item,
    NN_ACTIVATION_FUNCTION activation_function);

#endif //naive_implementations