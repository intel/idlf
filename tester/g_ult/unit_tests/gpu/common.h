/*
Copyright (c) 2015, Intel Corporation

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

#ifndef __DEVICE_GPU_COMMON__
#define __DEVICE_GPU_COMMON__
#include <stdint.h>
#include <memory>

#include "device/api/nn_device_api.h"
#include "device/common/nn_workload_data.h"
#include "device/api/nn_device_interface_0.h"
#include "device/gpu/api_internal/nn_device_interface_0_functions.h"

const int page_alignment = 4096;
typedef float (*fp_func_activ )( float );
typedef float (*fp_func_arithmetic )( float, float );
// Forward declaration
void generate_input_data( float *        &buffer,
                          uint_least32_t input_width,
                          uint_least32_t input_height,
                          uint_least32_t input_depth,
                          uint_least32_t num_inputs );
void generate_filter_data( float *        &filters,
                           uint_least32_t filter_width,
                           uint_least32_t filter_height,
                           uint_least32_t filter_depth,
                           uint_least32_t num_filters );
float mytanh( float val );
float relu( float val );
float none( float val );
float softplus( float val );

float none2f(float val1, float val2);
float add2f(float val1, float val2);
float sub2f(float val1, float val2);
float mul2f(float val1, float val2);
float div2f(float val1, float val2);

void arithmetic_ref( float                       *outputs,
                     const float          *const inputs,
                     const float          *const factor,
                     const uint_least32_t        num_input_feature_maps,
                     const uint_least32_t        input_feature_map_width,
                     const uint_least32_t        input_feature_map_height,
                     uint_least32_t              num_batches,
                     fp_func_arithmetic          FA );

void softmax_ref( float                     *outputs,
                  const float        *const inputs,
                  uint_least32_t            num_samples, 
                  uint_least32_t            num_batches );   
void normalize_ref( float                     *outputs,
                    const float               *const inputs,
                    uint_least32_t            num_batches,
                    const uint_least32_t      num_input_feature_maps,
                    const uint_least32_t      input_feature_map_width,
                    const uint_least32_t      input_feature_map_height,
                    const uint_least32_t      normalization_size,            // normalization area
                    const uint_least32_t      k,                             //hyper parameter k
                    const float               alpha,                //hyper parameter alpha
                    const float               beta,                 //hyper parameter k
                    NN_NORMALIZATION_MODE     normalization_mode ); // mode of naormalization

void view_ref( float                     *outputs,
               const float        *const inputs,
               nn_workload_data_coords_t          output_view_begin,
               nn_workload_data_coords_t          output_view_end,
               unsigned int output_width,
               unsigned int output_height,
               unsigned int output_depth);
void pool_ref( float                     *outputs,
               const float        *const inputs,
               nn_workload_data_coords_t          output_view_begin,
               nn_workload_data_coords_t          output_view_end,
               nn_workload_data_coords_t          input_view_begin,
               nn_workload_data_coords_t          input_view_end,
               const unsigned int        outputs_width,
               const unsigned int        outputs_height,
               const unsigned int        outputs_depth,
               const unsigned int        inputs_width,
               const unsigned int        inputs_height,
               const unsigned int        inputs_depth,
               const unsigned int        window_size,
               const unsigned int        stride_x,
               const unsigned int        stride_y,
               uint_least32_t            num_batches ); 
void fully_connect_ref( fp_func_activ             FA,
                        float                     *outputs,
                        const float        *const inputs,
                        const float        *const filters,
                        const float        *const biases,
                        const unsigned int        num_outputs,
                        const unsigned int        inputs_width,
                        const unsigned int        inputs_height,
                        const unsigned int        inputs_depth,
                        const unsigned int        num_batches );
void convolve_ref( fp_func_activ             FA,
                   float                     *outputs,
                   const float        *const inputs,
                   const float        *const filters,
                   const float        *const biases,
                   nn_workload_data_coords_t          output_view_begin,
                   nn_workload_data_coords_t          output_view_end,
                   nn_workload_data_coords_t          input_view_begin,
                   nn_workload_data_coords_t          input_view_end,
                   const unsigned int        outputs_width,
                   const unsigned int        outputs_height,
                   const unsigned int        outputs_depth,
                   const unsigned int        inputs_width,
                   const unsigned int        inputs_height,
                   const unsigned int        inputs_depth,
                   const unsigned int        weights_width,
                   const unsigned int        weights_height,
                   const unsigned int        weights_depth,
                   unsigned int              stridex,
                   unsigned int              stridey,
                   unsigned int              centerx,
                   unsigned int              centery,
                   const unsigned int        num_batches );

void init_data( float * &buffer, uint_least32_t bufferCount, float initValue );

std::unique_ptr< nn::workload_data<> > create_nn_workload_data_using_buffer( const float        *const buffer,
                                                                       nn_workload_data_layout_t          &buffer_layout,
                                                                       nn_workload_data_coords_t          &buffer_coords );
bool verify_output( std::unique_ptr<nn::data<float, 0>> &output_data, float * const ref_buf );
#endif
