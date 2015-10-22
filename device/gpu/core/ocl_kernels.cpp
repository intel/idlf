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
#include <string>
std::string conv_kernel1 = R"(

typedef float   real;
typedef float2  real2;
typedef float2  real3;
typedef float4  real4;
typedef float8  real8;
typedef float16 real16;


#define _IW INPUT_WIDTH
#define _IH INPUT_HEIGHT
#define _ID INPUT_DEPTH

#define _OW OUTPUT_WIDTH
#define _OH OUTPUT_HEIGHT
#define _OD NUM_FILTERS

#define FILTER_DEPTH INPUT_DEPTH
#define NUM_INPUT INPUT_DEPTH
#define NUM_OUTPUT NUM_FILTERS

#ifdef INCLUDE_generic_convolve
//TODO: Optimize following kernel


// OUTPUT_WIDTH, OUTPUT_HEIGHT
__kernel void generic_convolve(__global floatx* output, __global floatx* input, filter_qualifier floatx* filter, __global floatx* biases, unsigned int batch_output_offset)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int z = get_global_id(2);

    const unsigned int out_map_size = OUTPUT_WIDTH * OUTPUT_HEIGHT;

    const unsigned int filter_size = FILTER_DEPTH*FILTER_HEIGHT*FILTER_WIDTH;

    floatx dotProd  =  biases[z];

    unsigned int filter_offset = z * filter_size;
    unsigned int input_offset = STRIDEY * y * INPUT_WIDTH + x * STRIDEX;

    for (unsigned int k = 0; k < FILTER_DEPTH; ++k) {
        for (unsigned int j = 0; j < FILTER_HEIGHT; ++j) {
            for (unsigned int i = 0; i < FILTER_WIDTH; ++i) {
                floatx signal;
                signal = input[input_offset];
                dotProd += signal * filter[filter_offset];
                ++input_offset;
                ++filter_offset;
            }
            input_offset += INPUT_WIDTH - FILTER_WIDTH;
        }
        input_offset += (INPUT_HEIGHT - FILTER_HEIGHT)*INPUT_WIDTH;
    }

    output[batch_output_offset + z * out_map_size + y*OUTPUT_WIDTH + x] = activation_function(dotProd);

}
#endif
)";

std::string conv_kernel2a = R"(
#ifdef INCLUDE_convolve_5x5x1_v4x4x2_i_readInColumns

__attribute__((reqd_work_group_size(12, 6, 1)))
__kernel void convolve_5x5x1_v4x4x2_i_readInColumns(__global floatx* output, __global floatx* input, filter_qualifier floatx* weights, __global real* biases, unsigned int batch_output_offset)
{
    const unsigned output_stride = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    const unsigned output_fm = get_group_id(0) * 4;

    const unsigned x = get_local_id(0) * 2;
    const unsigned y = get_local_id(1) * 4;

    //Do four outputs vectorized across feature maps
    real4 top_outputs_l = 0;
    real4 middle1_outputs_l = 0;
    real4 middle2_outputs_l = 0;
    real4 bottom_outputs_l = 0;
    real4 top_outputs_r = 0;
    real4 middle1_outputs_r = 0;
    real4 middle2_outputs_r = 0;
    real4 bottom_outputs_r = 0;

    real16 input0, input1, input2, input3;
    real4 my_weights;

    __global real* my_input = input + y * INPUT_WIDTH + (x);        // 4 for kernel stride
    unsigned weight_idx = get_group_id(0) * 25;
    const unsigned weight_stride = 25 * NUM_OUTPUT / 4;         // 4 is for WEIGHT_INTERLIEVE=4

    for (unsigned input_fm = 0; input_fm < NUM_INPUT; ++input_fm)
    {
        input1 = vload16(0, my_input);
        input2 = vload16(0, my_input + INPUT_WIDTH);
        input3 = vload16(0, my_input + INPUT_WIDTH + INPUT_WIDTH);

        real4 weights_master0 = vload4(weight_idx + get_sub_group_local_id(), weights);
        real4 weights_master1 = vload4(weight_idx + 8 + get_sub_group_local_id(), weights);
        real4 weights_master2 = vload4(weight_idx + 16 + get_sub_group_local_id(), weights);
        real4 weights_master3 = vload4(weight_idx + 24 /*+ get_sub_group_local_id()*/, weights);

        {
            input0 = input1;
            input1 = input2;
            input2 = input3;
            input3 = vload16(0, my_input + (3 + 0)*INPUT_WIDTH);

            my_weights = intel_sub_group_shuffle(weights_master0, 0);
            top_outputs_l += input0.s0 * my_weights;
            middle1_outputs_l += input1.s0 * my_weights;
            middle2_outputs_l += input2.s0 * my_weights;
            bottom_outputs_l += input3.s0 * my_weights;

            top_outputs_r += input0.s1 * my_weights;
            middle1_outputs_r += input1.s1 * my_weights;
            middle2_outputs_r += input2.s1 * my_weights;
            bottom_outputs_r += input3.s1 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master0, 1);
            top_outputs_l += input0.s1 * my_weights;
            middle1_outputs_l += input1.s1 * my_weights;
            middle2_outputs_l += input2.s1 * my_weights;
            bottom_outputs_l += input3.s1 * my_weights;

            top_outputs_r += input0.s2 * my_weights;
            middle1_outputs_r += input1.s2 * my_weights;
            middle2_outputs_r += input2.s2 * my_weights;
            bottom_outputs_r += input3.s2 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master0, 2);
            top_outputs_l += input0.s2 * my_weights;
            middle1_outputs_l += input1.s2 * my_weights;
            middle2_outputs_l += input2.s2 * my_weights;
            bottom_outputs_l += input3.s2 * my_weights;

            top_outputs_r += input0.s3 * my_weights;
            middle1_outputs_r += input1.s3 * my_weights;
            middle2_outputs_r += input2.s3 * my_weights;
            bottom_outputs_r += input3.s3 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master0, 3);
            top_outputs_l += input0.s3 * my_weights;
            middle1_outputs_l += input1.s3 * my_weights;
            middle2_outputs_l += input2.s3 * my_weights;
            bottom_outputs_l += input3.s3 * my_weights;

            top_outputs_r += input0.s4 * my_weights;
            middle1_outputs_r += input1.s4 * my_weights;
            middle2_outputs_r += input2.s4 * my_weights;
            bottom_outputs_r += input3.s4 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master0, 4);
            top_outputs_l += input0.s4 * my_weights;
            middle1_outputs_l += input1.s4 * my_weights;
            middle2_outputs_l += input2.s4 * my_weights;
            bottom_outputs_l += input3.s4 * my_weights;

            top_outputs_r += input0.s5 * my_weights;
            middle1_outputs_r += input1.s5 * my_weights;
            middle2_outputs_r += input2.s5 * my_weights;
            bottom_outputs_r += input3.s5 * my_weights;

        }

        {
            input0 = input1;
            input1 = input2;
            input2 = input3;
            input3 = vload16(0, my_input + (3 + 1)*INPUT_WIDTH);

            my_weights = intel_sub_group_shuffle(weights_master0, 5);
            top_outputs_l += input0.s0 * my_weights;
            middle1_outputs_l += input1.s0 * my_weights;
            middle2_outputs_l += input2.s0 * my_weights;
            bottom_outputs_l += input3.s0 * my_weights;

            top_outputs_r += input0.s1 * my_weights;
            middle1_outputs_r += input1.s1 * my_weights;
            middle2_outputs_r += input2.s1 * my_weights;
            bottom_outputs_r += input3.s1 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master0, 6);
            top_outputs_l += input0.s1 * my_weights;
            middle1_outputs_l += input1.s1 * my_weights;
            middle2_outputs_l += input2.s1 * my_weights;
            bottom_outputs_l += input3.s1 * my_weights;

            top_outputs_r += input0.s2 * my_weights;
            middle1_outputs_r += input1.s2 * my_weights;
            middle2_outputs_r += input2.s2 * my_weights;
            bottom_outputs_r += input3.s2 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master0, 7);
            top_outputs_l += input0.s2 * my_weights;
            middle1_outputs_l += input1.s2 * my_weights;
            middle2_outputs_l += input2.s2 * my_weights;
            bottom_outputs_l += input3.s2 * my_weights;

            top_outputs_r += input0.s3 * my_weights;
            middle1_outputs_r += input1.s3 * my_weights;
            middle2_outputs_r += input2.s3 * my_weights;
            bottom_outputs_r += input3.s3 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master1, 0);
            top_outputs_l += input0.s3 * my_weights;
            middle1_outputs_l += input1.s3 * my_weights;
            middle2_outputs_l += input2.s3 * my_weights;
            bottom_outputs_l += input3.s3 * my_weights;

            top_outputs_r += input0.s4 * my_weights;
            middle1_outputs_r += input1.s4 * my_weights;
            middle2_outputs_r += input2.s4 * my_weights;
            bottom_outputs_r += input3.s4 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master1, 1);
            top_outputs_l += input0.s4 * my_weights;
            middle1_outputs_l += input1.s4 * my_weights;
            middle2_outputs_l += input2.s4 * my_weights;
            bottom_outputs_l += input3.s4 * my_weights;

            top_outputs_r += input0.s5 * my_weights;
            middle1_outputs_r += input1.s5 * my_weights;
            middle2_outputs_r += input2.s5 * my_weights;
            bottom_outputs_r += input3.s5 * my_weights;

        }

        {
            input0 = input1;
            input1 = input2;
            input2 = input3;
            input3 = vload16(0, my_input + (3 + 2)*INPUT_WIDTH);

            my_weights = intel_sub_group_shuffle(weights_master1, 2);
            top_outputs_l += input0.s0 * my_weights;
            middle1_outputs_l += input1.s0 * my_weights;
            middle2_outputs_l += input2.s0 * my_weights;
            bottom_outputs_l += input3.s0 * my_weights;

            top_outputs_r += input0.s1 * my_weights;
            middle1_outputs_r += input1.s1 * my_weights;
            middle2_outputs_r += input2.s1 * my_weights;
            bottom_outputs_r += input3.s1 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master1, 3);
            top_outputs_l += input0.s1 * my_weights;
            middle1_outputs_l += input1.s1 * my_weights;
            middle2_outputs_l += input2.s1 * my_weights;
            bottom_outputs_l += input3.s1 * my_weights;

            top_outputs_r += input0.s2 * my_weights;
            middle1_outputs_r += input1.s2 * my_weights;
            middle2_outputs_r += input2.s2 * my_weights;
            bottom_outputs_r += input3.s2 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master1, 4);
            top_outputs_l += input0.s2 * my_weights;
            middle1_outputs_l += input1.s2 * my_weights;
            middle2_outputs_l += input2.s2 * my_weights;
            bottom_outputs_l += input3.s2 * my_weights;

            top_outputs_r += input0.s3 * my_weights;
            middle1_outputs_r += input1.s3 * my_weights;
            middle2_outputs_r += input2.s3 * my_weights;
            bottom_outputs_r += input3.s3 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master1, 5);
            top_outputs_l += input0.s3 * my_weights;
            middle1_outputs_l += input1.s3 * my_weights;
            middle2_outputs_l += input2.s3 * my_weights;
            bottom_outputs_l += input3.s3 * my_weights;

            top_outputs_r += input0.s4 * my_weights;
            middle1_outputs_r += input1.s4 * my_weights;
            middle2_outputs_r += input2.s4 * my_weights;
            bottom_outputs_r += input3.s4 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master1, 6);
            top_outputs_l += input0.s4 * my_weights;
            middle1_outputs_l += input1.s4 * my_weights;
            middle2_outputs_l += input2.s4 * my_weights;
            bottom_outputs_l += input3.s4 * my_weights;

            top_outputs_r += input0.s5 * my_weights;
            middle1_outputs_r += input1.s5 * my_weights;
            middle2_outputs_r += input2.s5 * my_weights;
            bottom_outputs_r += input3.s5 * my_weights;

        }

        {
            input0 = input1;
            input1 = input2;
            input2 = input3;
            input3 = vload16(0, my_input + (3 + 3)*INPUT_WIDTH);

            my_weights = intel_sub_group_shuffle(weights_master1, 7);
            top_outputs_l += input0.s0 * my_weights;
            middle1_outputs_l += input1.s0 * my_weights;
            middle2_outputs_l += input2.s0 * my_weights;
            bottom_outputs_l += input3.s0 * my_weights;

            top_outputs_r += input0.s1 * my_weights;
            middle1_outputs_r += input1.s1 * my_weights;
            middle2_outputs_r += input2.s1 * my_weights;
            bottom_outputs_r += input3.s1 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master2, 0);
            top_outputs_l += input0.s1 * my_weights;
            middle1_outputs_l += input1.s1 * my_weights;
            middle2_outputs_l += input2.s1 * my_weights;
            bottom_outputs_l += input3.s1 * my_weights;

            top_outputs_r += input0.s2 * my_weights;
            middle1_outputs_r += input1.s2 * my_weights;
            middle2_outputs_r += input2.s2 * my_weights;
            bottom_outputs_r += input3.s2 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master2, 1);
            top_outputs_l += input0.s2 * my_weights;
            middle1_outputs_l += input1.s2 * my_weights;
            middle2_outputs_l += input2.s2 * my_weights;
            bottom_outputs_l += input3.s2 * my_weights;

            top_outputs_r += input0.s3 * my_weights;
            middle1_outputs_r += input1.s3 * my_weights;
            middle2_outputs_r += input2.s3 * my_weights;
            bottom_outputs_r += input3.s3 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master2, 2);
            top_outputs_l += input0.s3 * my_weights;
            middle1_outputs_l += input1.s3 * my_weights;
            middle2_outputs_l += input2.s3 * my_weights;
            bottom_outputs_l += input3.s3 * my_weights;

            top_outputs_r += input0.s4 * my_weights;
            middle1_outputs_r += input1.s4 * my_weights;
            middle2_outputs_r += input2.s4 * my_weights;
            bottom_outputs_r += input3.s4 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master2, 3);
            top_outputs_l += input0.s4 * my_weights;
            middle1_outputs_l += input1.s4 * my_weights;
            middle2_outputs_l += input2.s4 * my_weights;
            bottom_outputs_l += input3.s4 * my_weights;

            top_outputs_r += input0.s5 * my_weights;
            middle1_outputs_r += input1.s5 * my_weights;
            middle2_outputs_r += input2.s5 * my_weights;
            bottom_outputs_r += input3.s5 * my_weights;

        }

        {
            input0 = input1;
            input1 = input2;
            input2 = input3;
            input3 = vload16(0, my_input + (3 + 4)*INPUT_WIDTH);

            my_weights = intel_sub_group_shuffle(weights_master2, 4);
            top_outputs_l += input0.s0 * my_weights;
            middle1_outputs_l += input1.s0 * my_weights;
            middle2_outputs_l += input2.s0 * my_weights;
            bottom_outputs_l += input3.s0 * my_weights;

            top_outputs_r += input0.s1 * my_weights;
            middle1_outputs_r += input1.s1 * my_weights;
            middle2_outputs_r += input2.s1 * my_weights;
            bottom_outputs_r += input3.s1 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master2, 5);
            top_outputs_l += input0.s1 * my_weights;
            middle1_outputs_l += input1.s1 * my_weights;
            middle2_outputs_l += input2.s1 * my_weights;
            bottom_outputs_l += input3.s1 * my_weights;

            top_outputs_r += input0.s2 * my_weights;
            middle1_outputs_r += input1.s2 * my_weights;
            middle2_outputs_r += input2.s2 * my_weights;
            bottom_outputs_r += input3.s2 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master2, 6);
            top_outputs_l += input0.s2 * my_weights;
            middle1_outputs_l += input1.s2 * my_weights;
            middle2_outputs_l += input2.s2 * my_weights;
            bottom_outputs_l += input3.s2 * my_weights;

            top_outputs_r += input0.s3 * my_weights;
            middle1_outputs_r += input1.s3 * my_weights;
            middle2_outputs_r += input2.s3 * my_weights;
            bottom_outputs_r += input3.s3 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master2, 7);
            top_outputs_l += input0.s3 * my_weights;
            middle1_outputs_l += input1.s3 * my_weights;
            middle2_outputs_l += input2.s3 * my_weights;
            bottom_outputs_l += input3.s3 * my_weights;

            top_outputs_r += input0.s4 * my_weights;
            middle1_outputs_r += input1.s4 * my_weights;
            middle2_outputs_r += input2.s4 * my_weights;
            bottom_outputs_r += input3.s4 * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master3, 0);
            top_outputs_l += input0.s4 * my_weights;
            middle1_outputs_l += input1.s4 * my_weights;
            middle2_outputs_l += input2.s4 * my_weights;
            bottom_outputs_l += input3.s4 * my_weights;

            top_outputs_r += input0.s5 * my_weights;
            middle1_outputs_r += input1.s5 * my_weights;
            middle2_outputs_r += input2.s5 * my_weights;
            bottom_outputs_r += input3.s5 * my_weights;

        }

        weight_idx += weight_stride;
        my_input += INPUT_WIDTH * INPUT_HEIGHT;
    }
)";
// MSVC got limitation on std::string so we need to split that kernel in two kernels
std::string conv_kernel2b = R"(
    unsigned out_idx = batch_output_offset + output_fm * output_stride + y * OUTPUT_WIDTH + x;
    real bias = biases[output_fm];
    output[out_idx]                         = activation_function(bias + top_outputs_l.s0);
    output[out_idx + 1]                     = activation_function(bias + top_outputs_r.s0);
    output[out_idx + OUTPUT_WIDTH]          = activation_function(bias + middle1_outputs_l.s0);
    output[out_idx + OUTPUT_WIDTH + 1]      = activation_function(bias + middle1_outputs_r.s0);
    output[out_idx + 2 * OUTPUT_WIDTH]      = activation_function(bias + middle2_outputs_l.s0);
    output[out_idx + 2 * OUTPUT_WIDTH + 1]  = activation_function(bias + middle2_outputs_r.s0);
    output[out_idx + 3 * OUTPUT_WIDTH]      = activation_function(bias + bottom_outputs_l.s0);
    output[out_idx + 3 * OUTPUT_WIDTH + 1]  = activation_function(bias + bottom_outputs_r.s0);

    out_idx += output_stride;
    bias = biases[output_fm + 1];
    output[out_idx]                        = activation_function(bias  +  top_outputs_l.s1);
    output[out_idx + 1]                    = activation_function(bias  +  top_outputs_r.s1);
    output[out_idx + OUTPUT_WIDTH]         = activation_function(bias  +  middle1_outputs_l.s1);
    output[out_idx + OUTPUT_WIDTH + 1]     = activation_function(bias  +  middle1_outputs_r.s1);
    output[out_idx + 2 * OUTPUT_WIDTH]     = activation_function(bias  +  middle2_outputs_l.s1);
    output[out_idx + 2 * OUTPUT_WIDTH + 1] = activation_function(bias  +  middle2_outputs_r.s1);
    output[out_idx + 3 * OUTPUT_WIDTH]     = activation_function(bias  +  bottom_outputs_l.s1);
    output[out_idx + 3 * OUTPUT_WIDTH + 1] = activation_function(bias  +  bottom_outputs_r.s1);

    out_idx += output_stride;
    bias = biases[output_fm + 2];
    output[out_idx]                        = activation_function(bias + top_outputs_l.s2);
    output[out_idx + 1]                    = activation_function(bias + top_outputs_r.s2);
    output[out_idx + OUTPUT_WIDTH]         = activation_function(bias + middle1_outputs_l.s2);
    output[out_idx + OUTPUT_WIDTH + 1]     = activation_function(bias + middle1_outputs_r.s2);
    output[out_idx + 2 * OUTPUT_WIDTH]     = activation_function(bias + middle2_outputs_l.s2);
    output[out_idx + 2 * OUTPUT_WIDTH + 1] = activation_function(bias + middle2_outputs_r.s2);
    output[out_idx + 3 * OUTPUT_WIDTH]     = activation_function(bias + bottom_outputs_l.s2);
    output[out_idx + 3 * OUTPUT_WIDTH + 1] = activation_function(bias + bottom_outputs_r.s2);

    out_idx += output_stride;
    bias = biases[output_fm + 3];
    output[out_idx]                        = activation_function(bias + top_outputs_l.s3);
    output[out_idx + 1]                    = activation_function(bias + top_outputs_r.s3);
    output[out_idx + OUTPUT_WIDTH]         = activation_function(bias + middle1_outputs_l.s3);
    output[out_idx + OUTPUT_WIDTH + 1]     = activation_function(bias + middle1_outputs_r.s3);
    output[out_idx + 2 * OUTPUT_WIDTH]     = activation_function(bias + middle2_outputs_l.s3);
    output[out_idx + 2 * OUTPUT_WIDTH + 1] = activation_function(bias + middle2_outputs_r.s3);
    output[out_idx + 3 * OUTPUT_WIDTH]     = activation_function(bias + bottom_outputs_l.s3);
    output[out_idx + 3 * OUTPUT_WIDTH + 1] = activation_function(bias + bottom_outputs_r.s3);
}
#endif  // INCLUDE_convolve_5x5x1_v4x4x2_i_readInColumns
)";

std::string conv_kernel3a = R"(
#ifdef INCLUDE_convolve_3x3x1_v8x3x3_i_readInColumns

#ifndef OWPAD
#define OWPAD  0
#endif

#ifndef OHPAD
#define OHPAD  0
#endif

#ifndef GROUPING
#define GROUPING 8
#endif

#if GROUPING == 16
#define realX real16
#define vloadX vload16
#endif

#if GROUPING == 8
#define realX real8
#define vloadX vload8
#endif

#if GROUPING == 4
#define realX real4
#define vloadX vload4
#endif

#if GROUPING == 2
#define realX real2
#define vloadX vload2
#endif

#ifndef DISABLE_LOCAL_GROUPS
__attribute__((reqd_work_group_size(LWS_X, LWS_Y, 1)))
#endif //#ifndef DISABLE_LOCAL_GROUPS
__kernel void convolve_3x3x1_v8x3x3_i_readInColumns(__global real* output, __global real* input, filter_qualifier real* weights, __global real* biases, unsigned int batch_output_offset)
{
    const unsigned int OUTPUT_WIDTH_PADDED =  OUTPUT_WIDTH  + OWPAD;
    const unsigned int OUTPUT_HEIGHT_PADDED = OUTPUT_HEIGHT + OHPAD;
    const unsigned output_stride = OUTPUT_WIDTH_PADDED * OUTPUT_HEIGHT_PADDED;

#ifndef DISABLE_LOCAL_GROUPS
    const unsigned output_fm     = get_group_id(0) * GROUPING;  // 8 ofm in one work item
    const unsigned x = get_local_id(0) * 3;   // 3x3 outputs in one work item
    const unsigned y = get_local_id(1) * 3;
    unsigned weight_idx      = get_group_id(0) * 9;
#else //#ifndef DISABLE_LOCAL_GROUPS
    const unsigned output_fm     = get_global_id(2) * GROUPING;  // 8 ofm in one work item
    const unsigned x = get_global_id(0) * 3;   // 3x3 outputs in one work item
    const unsigned y = get_global_id(1) * 3;
    unsigned weight_idx      = get_global_id(2) * 9;
#endif ///#ifndef DISABLE_LOCAL_GROUPS

    //Do four outputs vectorized across feature maps
    realX top_outputs_l    = 0;
    realX middle_outputs_l = 0;
    realX bottom_outputs_l = 0;

    realX top_outputs_m    = 0;
    realX middle_outputs_m = 0;
    realX bottom_outputs_m = 0;

    realX top_outputs_r    = 0;
    realX middle_outputs_r = 0;
    realX bottom_outputs_r = 0;

    realX my_weights;
    real8 input_up, input_mid, input_down;

    __global real* my_input = input + y * INPUT_WIDTH + x;
    const unsigned weight_stride = 9 * NUM_OUTPUT / GROUPING;           // "/ GROUPING" is for WEIGHT_INTERLIEVE=GROUPING

    for (unsigned input_fm = 0; input_fm < NUM_INPUT; ++input_fm)
    {
#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx, weights);
#else
        realX weights_master0 = vloadX(weight_idx + get_sub_group_local_id(), weights);

        my_weights  = intel_sub_group_shuffle(weights_master0, 0);
#endif

        input_up        = vload8(0, my_input);
        input_mid       = vload8(0, my_input + INPUT_WIDTH);
        input_down      = vload8(0, my_input + 2 * INPUT_WIDTH);

        top_outputs_l    += input_up.s0   * my_weights;
        middle_outputs_l += input_mid.s0  * my_weights;
        bottom_outputs_l += input_down.s0 * my_weights;
        top_outputs_m    += input_up.s1   * my_weights;
        middle_outputs_m += input_mid.s1  * my_weights;
        bottom_outputs_m += input_down.s1 * my_weights;
        top_outputs_r    += input_up.s2   * my_weights;
        middle_outputs_r += input_mid.s2  * my_weights;
        bottom_outputs_r += input_down.s2 * my_weights;

#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx+1, weights);
#else
        my_weights  = intel_sub_group_shuffle(weights_master0, 1);
#endif
        top_outputs_l    += input_up.s1   * my_weights;
        middle_outputs_l += input_mid.s1  * my_weights;
        bottom_outputs_l += input_down.s1 * my_weights;
        top_outputs_m    += input_up.s2   * my_weights;
        middle_outputs_m += input_mid.s2  * my_weights;
        bottom_outputs_m += input_down.s2 * my_weights;
        top_outputs_r    += input_up.s3   * my_weights;
        middle_outputs_r += input_mid.s3  * my_weights;
        bottom_outputs_r += input_down.s3 * my_weights;

#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx+2, weights);
#else
        my_weights  = intel_sub_group_shuffle(weights_master0, 2);
#endif
        top_outputs_l    += input_up.s2   * my_weights;
        middle_outputs_l += input_mid.s2  * my_weights;
        bottom_outputs_l += input_down.s2 * my_weights;
        top_outputs_m    += input_up.s3   * my_weights;
        middle_outputs_m += input_mid.s3  * my_weights;
        bottom_outputs_m += input_down.s3 * my_weights;
        top_outputs_r    += input_up.s4   * my_weights;
        middle_outputs_r += input_mid.s4  * my_weights;
        bottom_outputs_r += input_down.s4 * my_weights;

        input_up        = input_mid;
        input_mid       = input_down;

        input_down      = vload8(0, my_input + 3 * INPUT_WIDTH);

#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx+3, weights);
#else
        my_weights  = intel_sub_group_shuffle(weights_master0, 3);
#endif
        top_outputs_l    += input_up.s0   * my_weights;
        middle_outputs_l += input_mid.s0  * my_weights;
        bottom_outputs_l += input_down.s0 * my_weights;
        top_outputs_m    += input_up.s1   * my_weights;
        middle_outputs_m += input_mid.s1  * my_weights;
        bottom_outputs_m += input_down.s1 * my_weights;
        top_outputs_r    += input_up.s2   * my_weights;
        middle_outputs_r += input_mid.s2  * my_weights;
        bottom_outputs_r += input_down.s2 * my_weights;

#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx+4, weights);
#else
        my_weights  = intel_sub_group_shuffle(weights_master0, 4);
#endif
        top_outputs_l    += input_up.s1   * my_weights;
        middle_outputs_l += input_mid.s1  * my_weights;
        bottom_outputs_l += input_down.s1 * my_weights;
        top_outputs_m    += input_up.s2   * my_weights;
        middle_outputs_m += input_mid.s2  * my_weights;
        bottom_outputs_m += input_down.s2 * my_weights;
        top_outputs_r    += input_up.s3   * my_weights;
        middle_outputs_r += input_mid.s3  * my_weights;
        bottom_outputs_r += input_down.s3 * my_weights;

#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx+5, weights);
#else
        my_weights  = intel_sub_group_shuffle(weights_master0, 5);
#endif
        top_outputs_l    += input_up.s2   * my_weights;
        middle_outputs_l += input_mid.s2  * my_weights;
        bottom_outputs_l += input_down.s2 * my_weights;
        top_outputs_m    += input_up.s3   * my_weights;
        middle_outputs_m += input_mid.s3  * my_weights;
        bottom_outputs_m += input_down.s3 * my_weights;
        top_outputs_r    += input_up.s4   * my_weights;
        middle_outputs_r += input_mid.s4  * my_weights;
        bottom_outputs_r += input_down.s4 * my_weights;

        input_up        = input_mid;
        input_mid       = input_down;

        input_down      = vload8(0, my_input + 4 * INPUT_WIDTH);

#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx+6, weights);
#else
        my_weights  = intel_sub_group_shuffle(weights_master0, 6);
#endif
        top_outputs_l    += input_up.s0   * my_weights;
        middle_outputs_l += input_mid.s0  * my_weights;
        bottom_outputs_l += input_down.s0 * my_weights;
        top_outputs_m    += input_up.s1   * my_weights;
        middle_outputs_m += input_mid.s1  * my_weights;
        bottom_outputs_m += input_down.s1 * my_weights;
        top_outputs_r    += input_up.s2   * my_weights;
        middle_outputs_r += input_mid.s2  * my_weights;
        bottom_outputs_r += input_down.s2 * my_weights;

#ifdef DISABLE_SIMD_SHUFFLE
        my_weights = vloadX(weight_idx+7, weights);
#else
        my_weights  = intel_sub_group_shuffle(weights_master0, 7);
#endif
        top_outputs_l    += input_up.s1   * my_weights;
        middle_outputs_l += input_mid.s1  * my_weights;
        bottom_outputs_l += input_down.s1 * my_weights;
        top_outputs_m    += input_up.s2   * my_weights;
        middle_outputs_m += input_mid.s2  * my_weights;
        bottom_outputs_m += input_down.s2 * my_weights;
        top_outputs_r    += input_up.s3   * my_weights;
        middle_outputs_r += input_mid.s3  * my_weights;
        bottom_outputs_r += input_down.s3 * my_weights;

        //my_weights  = intel_sub_group_shuffle(weights_master1, 0);  // TODO: why shuffle?
        my_weights  = vloadX(weight_idx + 8, weights);
        top_outputs_l    += input_up.s2   * my_weights;
        middle_outputs_l += input_mid.s2  * my_weights;
        bottom_outputs_l += input_down.s2 * my_weights;
        top_outputs_m    += input_up.s3   * my_weights;
        middle_outputs_m += input_mid.s3  * my_weights;
        bottom_outputs_m += input_down.s3 * my_weights;
        top_outputs_r    += input_up.s4   * my_weights;
        middle_outputs_r += input_mid.s4  * my_weights;
        bottom_outputs_r += input_down.s4 * my_weights;

        my_input       += INPUT_WIDTH * INPUT_HEIGHT;
        weight_idx     += weight_stride;
    }

    unsigned out_idx = OUT_BUFF_OFFSET + batch_output_offset + output_fm * output_stride + y * OUTPUT_WIDTH_PADDED + x;
    real bias = biases[output_fm];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s0);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s0);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s0);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s0);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s0);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s0);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s0);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s0);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s0);
    out_idx += output_stride;
    bias = biases[output_fm+1];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s1);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s1);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s1);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s1);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s1);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s1);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s1);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s1);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s1);

#if GROUPING >= 4
    out_idx += output_stride;
    bias = biases[output_fm+2];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s2);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s2);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s2);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s2);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s2);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s2);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s2);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s2);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s2);

    out_idx += output_stride;
    bias = biases[output_fm+3];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s3);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s3);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s3);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s3);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s3);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s3);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s3);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s3);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s3);
#endif //#if GROUPING >= 4

)";

std::string conv_kernel3b = R"(
#if GROUPING >= 8
    out_idx += output_stride;
    bias = biases[output_fm+4];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s4);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s4);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s4);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s4);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s4);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s4);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s4);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s4);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s4);

    out_idx += output_stride;
    bias = biases[output_fm+5];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s5);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s5);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s5);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s5);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s5);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s5);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s5);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s5);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s5);

    out_idx += output_stride;
    bias = biases[output_fm+6];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s6);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s6);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s6);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s6);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s6);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s6);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s6);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s6);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s6);

    out_idx += output_stride;
    bias = biases[output_fm+7];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s7);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s7);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s7);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s7);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s7);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s7);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s7);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s7);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s7);
#endif //#if GROUPING >= 8

#if GROUPING == 16
    out_idx += output_stride;
    bias = biases[output_fm+8];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s8);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s8);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s8);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s8);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s8);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s8);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s8);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s8);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s8);

    out_idx += output_stride;
    bias = biases[output_fm+9];
    output[out_idx]                                = activation_function( bias + top_outputs_l.s9);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.s9);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.s9);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.s9);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.s9);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.s9);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.s9);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.s9);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.s9);

    out_idx += output_stride;
    bias = biases[output_fm+10];
    output[out_idx]                                = activation_function( bias + top_outputs_l.sa);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.sa);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.sa);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.sa);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.sa);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.sa);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.sa);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.sa);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.sa);

    out_idx += output_stride;
    bias = biases[output_fm+11];
    output[out_idx]                                = activation_function( bias + top_outputs_l.sb);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.sb);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.sb);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.sb);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.sb);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.sb);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.sb);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.sb);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.sb);

    out_idx += output_stride;
    bias = biases[output_fm+12];
    output[out_idx]                                = activation_function( bias + top_outputs_l.sc);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.sc);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.sc);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.sc);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.sc);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.sc);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.sc);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.sc);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.sc);

    out_idx += output_stride;
    bias = biases[output_fm+13];
    output[out_idx]                                = activation_function( bias + top_outputs_l.sd);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.sd);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.sd);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.sd);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.sd);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.sd);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.sd);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.sd);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.sd);

    out_idx += output_stride;
    bias = biases[output_fm+14];
    output[out_idx]                                = activation_function( bias + top_outputs_l.se);
    output[out_idx + 1]                            = activation_function( bias + top_outputs_m.se);
    output[out_idx + 2]                            = activation_function( bias + top_outputs_r.se);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias + middle_outputs_l.se);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias + middle_outputs_m.se);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias + middle_outputs_r.se);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias + bottom_outputs_l.se);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias + bottom_outputs_m.se);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias + bottom_outputs_r.se);

    out_idx += output_stride;
    bias = biases[output_fm+15];
    output[out_idx]                                = activation_function( bias  + top_outputs_l.sf);
    output[out_idx + 1]                            = activation_function( bias  + top_outputs_m.sf);
    output[out_idx + 2]                            = activation_function( bias  + top_outputs_r.sf);
    output[out_idx + OUTPUT_WIDTH_PADDED]          = activation_function( bias  + middle_outputs_l.sf);
    output[out_idx + OUTPUT_WIDTH_PADDED + 1]      = activation_function( bias  + middle_outputs_m.sf);
    output[out_idx + OUTPUT_WIDTH_PADDED + 2]      = activation_function( bias  + middle_outputs_r.sf);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED]      = activation_function( bias  + bottom_outputs_l.sf);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 1]  = activation_function( bias  + bottom_outputs_m.sf);
    output[out_idx + 2 * OUTPUT_WIDTH_PADDED + 2]  = activation_function( bias  + bottom_outputs_r.sf);
#endif //#if GROUPING == 16
}
#endif // INCLUDE_convolve_3x3x1_v8x3x3_i_readInColumns
)";


std::string conv_kernel4 = R"(
#ifdef INCLUDE_convolve_11x11x4_v4x2x4_i

__attribute__((reqd_work_group_size(14, 28, 1)))
__kernel void convolve_11x11x4_v4x2x4_i(__global real* output, __global real* input, filter_qualifier real* weights, __global real* biases, unsigned int batch_output_offset)
{
    const unsigned output_stride = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    const unsigned output_fm     = get_group_id(0) * 4;

    const unsigned x = get_local_id(0) * 4;
    const unsigned y = get_local_id(1) * 2;

    real4 top_outputs_l             = 0;
    real4 bottom_outputs_l          = 0;
    real4 top_outputs_ml            = 0;
    real4 bottom_outputs_ml         = 0;
    real4 top_outputs_mr            = 0;
    real4 bottom_outputs_mr         = 0;
    real4 top_outputs_r             = 0;
    real4 bottom_outputs_r          = 0;

    real16 input0_l, input1_l;
    real8  input0_r, input1_r;

    real4 my_weights, weights_master;
    real2 weights_master1;

    __global real* my_input  = input   + (y * 4) * INPUT_WIDTH + (x * 4);       // 4 for kernel stride


    unsigned weight_idx      = get_group_id(0) * NUM_INPUT * 121;
    for (unsigned input_fm = 0; input_fm < NUM_INPUT; ++input_fm)
    {

        for(int i=0; i < 11; ++i)
        {
            weights_master        = vload4(weight_idx+get_sub_group_local_id(), weights);
            // I don't know it's legal, but seems to be a bit faster
            weights_master1       = vload2(2*weight_idx+get_sub_group_local_id()+16, weights);

            my_weights = intel_sub_group_shuffle(weights_master, 0);

            input0_l          = vload16(0, my_input + i * INPUT_WIDTH );
            input0_r          = vload8(0, my_input + i * INPUT_WIDTH + 16 );
            input1_l          = vload16(0, my_input + (4 + i) * INPUT_WIDTH );
            input1_r          = vload8(0, my_input + (4 + i) * INPUT_WIDTH + 16 );

            top_outputs_l       += input0_l.s0 * my_weights;
            bottom_outputs_l    += input1_l.s0 * my_weights;

            top_outputs_ml      += input0_l.s4 * my_weights;
            bottom_outputs_ml   += input1_l.s4 * my_weights;

            top_outputs_mr      += input0_l.s8 * my_weights;
            bottom_outputs_mr   += input1_l.s8 * my_weights;

            top_outputs_r       += input0_l.sc * my_weights;
            bottom_outputs_r    += input1_l.sc * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master, 1);
            top_outputs_l       += input0_l.s1 * my_weights;
            bottom_outputs_l    += input1_l.s1 * my_weights;

            top_outputs_ml      += input0_l.s5 * my_weights;
            bottom_outputs_ml   += input1_l.s5 * my_weights;

            top_outputs_mr      += input0_l.s9 * my_weights;
            bottom_outputs_mr   += input1_l.s9 * my_weights;

            top_outputs_r       += input0_l.sd * my_weights;
            bottom_outputs_r    += input1_l.sd * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master, 2);
            top_outputs_l       += input0_l.s2 * my_weights;
            bottom_outputs_l    += input1_l.s2 * my_weights;

            top_outputs_ml      += input0_l.s6 * my_weights;
            bottom_outputs_ml   += input1_l.s6 * my_weights;

            top_outputs_mr      += input0_l.sa * my_weights;
            bottom_outputs_mr   += input1_l.sa * my_weights;

            top_outputs_r       += input0_l.se * my_weights;
            bottom_outputs_r    += input1_l.se * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master, 3);
            top_outputs_l       += input0_l.s3 * my_weights;
            bottom_outputs_l    += input1_l.s3 * my_weights;

            top_outputs_ml      += input0_l.s7 * my_weights;
            bottom_outputs_ml   += input1_l.s7 * my_weights;

            top_outputs_mr      += input0_l.sb * my_weights;
            bottom_outputs_mr   += input1_l.sb * my_weights;

            top_outputs_r       += input0_l.sf * my_weights;
            bottom_outputs_r    += input1_l.sf * my_weights;

            my_weights = intel_sub_group_shuffle(weights_master, 4);
            top_outputs_l       += input0_l.s4 * my_weights;
            bottom_outputs_l    += input1_l.s4 * my_weights;

            top_outputs_ml      += input0_l.s8 * my_weights;
            bottom_outputs_ml   += input1_l.s8 * my_weights;

            top_outputs_mr      += input0_l.sc * my_weights;
            bottom_outputs_mr   += input1_l.sc * my_weights;

            top_outputs_r       += input0_r.s0 * my_weights;
            bottom_outputs_r    += input1_r.s0 * my_weights;


            my_weights = intel_sub_group_shuffle(weights_master, 5);
            top_outputs_l       += input0_l.s5 * my_weights;
            bottom_outputs_l    += input1_l.s5 * my_weights;

            top_outputs_ml      += input0_l.s9 * my_weights;
            bottom_outputs_ml   += input1_l.s9 * my_weights;

            top_outputs_mr      += input0_l.sd * my_weights;
            bottom_outputs_mr   += input1_l.sd * my_weights;

            top_outputs_r       += input0_r.s1 * my_weights;
            bottom_outputs_r    += input1_r.s1 * my_weights;


            my_weights = intel_sub_group_shuffle(weights_master, 6);
            top_outputs_l       += input0_l.s6 * my_weights;
            bottom_outputs_l    += input1_l.s6 * my_weights;

            top_outputs_ml      += input0_l.sa * my_weights;
            bottom_outputs_ml   += input1_l.sa * my_weights;

            top_outputs_mr      += input0_l.se * my_weights;
            bottom_outputs_mr   += input1_l.se * my_weights;

            top_outputs_r       += input0_r.s2 * my_weights;
            bottom_outputs_r    += input1_r.s2 * my_weights;


            my_weights = intel_sub_group_shuffle(weights_master, 7);
            top_outputs_l       += input0_l.s7 * my_weights;
            bottom_outputs_l    += input1_l.s7 * my_weights;

            top_outputs_ml      += input0_l.sb * my_weights;
            bottom_outputs_ml   += input1_l.sb * my_weights;

            top_outputs_mr      += input0_l.sf * my_weights;
            bottom_outputs_mr   += input1_l.sf * my_weights;

            top_outputs_r       += input0_r.s3 * my_weights;
            bottom_outputs_r    += input1_r.s3 * my_weights;


            //my_weights      = vload4(weight_idx + 8, weights);
            my_weights = (real4)(intel_sub_group_shuffle(weights_master1, 0), intel_sub_group_shuffle(weights_master1, 1));
            top_outputs_l       += input0_l.s8 * my_weights;
            bottom_outputs_l    += input1_l.s8 * my_weights;

            top_outputs_ml      += input0_l.sc * my_weights;
            bottom_outputs_ml   += input1_l.sc * my_weights;

            top_outputs_mr      += input0_r.s0 * my_weights;
            bottom_outputs_mr   += input1_r.s0 * my_weights;

            top_outputs_r       += input0_r.s4 * my_weights;
            bottom_outputs_r    += input1_r.s4 * my_weights;

            //my_weights      = vload4(weight_idx + 9, weights);
            my_weights = (real4)(intel_sub_group_shuffle(weights_master1, 2), intel_sub_group_shuffle(weights_master1, 3));
            top_outputs_l       += input0_l.s9 * my_weights;
            bottom_outputs_l    += input1_l.s9 * my_weights;

            top_outputs_ml      += input0_l.sd * my_weights;
            bottom_outputs_ml   += input1_l.sd * my_weights;

            top_outputs_mr      += input0_r.s1 * my_weights;
            bottom_outputs_mr   += input1_r.s1 * my_weights;

            top_outputs_r       += input0_r.s5 * my_weights;
            bottom_outputs_r    += input1_r.s5 * my_weights;


            //my_weights      = vload4(weight_idx + 10, weights);
            my_weights = (real4)(intel_sub_group_shuffle(weights_master1, 4), intel_sub_group_shuffle(weights_master1, 5));
            top_outputs_l       += input0_l.sa * my_weights;
            bottom_outputs_l    += input1_l.sa * my_weights;

            top_outputs_ml      += input0_l.se * my_weights;
            bottom_outputs_ml   += input1_l.se * my_weights;

            top_outputs_mr      += input0_r.s2 * my_weights;
            bottom_outputs_mr   += input1_r.s2 * my_weights;

            top_outputs_r       += input0_r.s6 * my_weights;
            bottom_outputs_r    += input1_r.s6 * my_weights;


            weight_idx += 11;
        }

        my_input   += INPUT_WIDTH * INPUT_HEIGHT;
    }

    unsigned out_idx = batch_output_offset + output_fm * output_stride + y * OUTPUT_WIDTH + x;
    real bias = biases[output_fm];
    output[out_idx]                             = activation_function(bias + top_outputs_l.s0);
    output[out_idx + 1]                         = activation_function(bias + top_outputs_ml.s0);
    output[out_idx + 2]                         = activation_function(bias + top_outputs_mr.s0);
    output[out_idx + 3]                         = activation_function(bias + top_outputs_r.s0);
    output[out_idx + OUTPUT_WIDTH]              = activation_function(bias + bottom_outputs_l.s0);
    output[out_idx + OUTPUT_WIDTH + 1]          = activation_function(bias + bottom_outputs_ml.s0);
    output[out_idx + OUTPUT_WIDTH + 2]          = activation_function(bias + bottom_outputs_mr.s0);
    output[out_idx + OUTPUT_WIDTH + 3]          = activation_function(bias + bottom_outputs_r.s0);

    out_idx += output_stride;
    bias = biases[output_fm + 1];
    output[out_idx]                             = activation_function(bias + top_outputs_l.s1);
    output[out_idx + 1]                         = activation_function(bias + top_outputs_ml.s1);
    output[out_idx + 2]                         = activation_function(bias + top_outputs_mr.s1);
    output[out_idx + 3]                         = activation_function(bias + top_outputs_r.s1);
    output[out_idx + OUTPUT_WIDTH]              = activation_function(bias + bottom_outputs_l.s1);
    output[out_idx + OUTPUT_WIDTH + 1]          = activation_function(bias + bottom_outputs_ml.s1);
    output[out_idx + OUTPUT_WIDTH + 2]          = activation_function(bias + bottom_outputs_mr.s1);
    output[out_idx + OUTPUT_WIDTH + 3]          = activation_function(bias + bottom_outputs_r.s1);

    out_idx += output_stride;
    bias = biases[output_fm + 2];
    output[out_idx]                             = activation_function(bias + top_outputs_l.s2);
    output[out_idx + 1]                         = activation_function(bias + top_outputs_ml.s2);
    output[out_idx + 2]                         = activation_function(bias + top_outputs_mr.s2);
    output[out_idx + 3]                         = activation_function(bias + top_outputs_r.s2);
    output[out_idx + OUTPUT_WIDTH]              = activation_function(bias + bottom_outputs_l.s2);
    output[out_idx + OUTPUT_WIDTH + 1]          = activation_function(bias + bottom_outputs_ml.s2);
    output[out_idx + OUTPUT_WIDTH + 2]          = activation_function(bias + bottom_outputs_mr.s2);
    output[out_idx + OUTPUT_WIDTH + 3]          = activation_function(bias + bottom_outputs_r.s2);

    out_idx += output_stride;
    bias = biases[output_fm + 3];
    output[out_idx]                             = activation_function(bias + top_outputs_l.s3);
    output[out_idx + 1]                         = activation_function(bias + top_outputs_ml.s3);
    output[out_idx + 2]                         = activation_function(bias + top_outputs_mr.s3);
    output[out_idx + 3]                         = activation_function(bias + top_outputs_r.s3);
    output[out_idx + OUTPUT_WIDTH]              = activation_function(bias + bottom_outputs_l.s3);
    output[out_idx + OUTPUT_WIDTH + 1]          = activation_function(bias + bottom_outputs_ml.s3);
    output[out_idx + OUTPUT_WIDTH + 2]          = activation_function(bias + bottom_outputs_mr.s3);
    output[out_idx + OUTPUT_WIDTH + 3]          = activation_function(bias + bottom_outputs_r.s3);

}

#endif //INCLUDE_convolve_11x11x4_v4x2x4_i
)";

std::string conv_kernel5 = R"(
#ifdef INCLUDE_convolve_6x6x1_v4x1_i_readInColumns
#ifdef REQD_WORK_GROUP_SIZE
__attribute__((reqd_work_group_size(256, 1, 1)))
#endif //REQD_WORK_GROUP_SIZE
__kernel void convolve_6x6x1_v4x1_i_readInColumns(__global real* output, __global real* input, filter_qualifier real* weights0, __global real* biases, unsigned int batch_output_offset, filter_qualifier real* weights1)
{
    const unsigned output_stride = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    const unsigned gid = get_global_id(0);
    const unsigned output_fm     = gid * 4;

    //Do four outputs vectorized across feature maps
    real4 outputs    = 0;
    real8 inputs;
    real4 my_weights;

    unsigned weight_idx = 36 * gid;
    const unsigned weight_stride = 36 * NUM_OUTPUT / 4;         // "/ 4" is for WEIGHT_INTERLIEVE=4

    __global real* my_input  = input;
    for (unsigned input_fm = 0; input_fm < NUM_INPUT/2; ++input_fm)
    {
        my_weights      = vload4(weight_idx, weights0);

        inputs    = vload8(0, my_input);


        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 1, weights0);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 2, weights0);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 3, weights0);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 4, weights0);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 5, weights0);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 6, weights0);
        inputs          = vload8(0, my_input + INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 7, weights0);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 8, weights0);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 9, weights0);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 10, weights0);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 11, weights0);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 12, weights0);
        inputs          = vload8(0, my_input + 2 * INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 13, weights0);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 14, weights0);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 15, weights0);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 16, weights0);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 17, weights0);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 18, weights0);
        inputs          = vload8(0, my_input + 3 * INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 19, weights0);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 20, weights0);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 21, weights0);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 22, weights0);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 23, weights0);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 24, weights0);
        inputs          = vload8(0, my_input + 4 * INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 25, weights0);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 26, weights0);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 27, weights0);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 28, weights0);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 29, weights0);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 30, weights0);
        inputs          = vload8(0, my_input + 5 * INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 31, weights0);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 32, weights0);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 33, weights0);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 34, weights0);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 35, weights0);
        outputs    += inputs.s5 * my_weights;

        my_input   += INPUT_WIDTH * INPUT_HEIGHT;
        weight_idx += weight_stride;
    }

    weight_idx = 36 * gid;
    for (unsigned input_fm =NUM_INPUT/2; input_fm < NUM_INPUT; ++input_fm)
    {
        my_weights      = vload4(weight_idx, weights1);
        inputs          = vload8(0, my_input);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 1, weights1);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 2, weights1);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 3, weights1);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 4, weights1);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 5, weights1);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 6, weights1);
        inputs          = vload8(0, my_input + INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 7, weights1);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 8, weights1);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 9, weights1);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 10, weights1);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 11, weights1);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 12, weights1);
        inputs          = vload8(0, my_input + 2 * INPUT_WIDTH);


        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 13, weights1);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 14, weights1);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 15, weights1);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 16, weights1);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 17, weights1);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 18, weights1);
        inputs          = vload8(0, my_input + 3 * INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 19, weights1);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 20, weights1);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 21, weights1);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 22, weights1);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 23, weights1);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 24, weights1);
        inputs          = vload8(0, my_input + 4 * INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 25, weights1);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 26, weights1);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 27, weights1);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 28, weights1);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 29, weights1);
        outputs    += inputs.s5 * my_weights;

        my_weights      = vload4(weight_idx + 30, weights1);
        inputs          = vload8(0, my_input + 5 * INPUT_WIDTH);

        outputs    += inputs.s0 * my_weights;

        my_weights      = vload4(weight_idx + 31, weights1);
        outputs    += inputs.s1 * my_weights;

        my_weights      = vload4(weight_idx + 32, weights1);
        outputs    += inputs.s2 * my_weights;

        my_weights      = vload4(weight_idx + 33, weights1);
        outputs    += inputs.s3 * my_weights;

        my_weights      = vload4(weight_idx + 34, weights1);
        outputs    += inputs.s4 * my_weights;

        my_weights      = vload4(weight_idx + 35, weights1);
        outputs    += inputs.s5 * my_weights;

        my_input   += INPUT_WIDTH * INPUT_HEIGHT;
        weight_idx += weight_stride;
    }

    unsigned out_idx = batch_output_offset + output_fm * output_stride;

    output[out_idx]                     = activation_function(biases[output_fm] + outputs.s0);
    out_idx += output_stride;

    output[out_idx]                     = activation_function(biases[output_fm + 1] + outputs.s1);
    out_idx += output_stride;

    output[out_idx]                     = activation_function(biases[output_fm + 2] + outputs.s2);
    out_idx += output_stride;

    output[out_idx]                     = activation_function(biases[output_fm + 3] + outputs.s3);
}
#endif //INCLUDE_convolve_6x6x1_v4x1_i_readInColumns
)";

std::string conv_kernel6 = R"(
#ifdef INCLUDE_convolve_6x6x1_v4x1_i_readInColumns_batch_loop
#ifdef REQD_WORK_GROUP_SIZE
__attribute__((reqd_work_group_size(LWS_X, 1, 1)))
#endif //REQD_WORK_GROUP_SIZE
__kernel void convolve_6x6x1_v4x1_i_readInColumns_batch_loop(__global real* output, __global real* input, filter_qualifier real* weights0, unsigned int batch_output_offset, filter_qualifier real* weights1)
{
    const unsigned output_stride = OUTPUT_WIDTH * OUTPUT_HEIGHT;

    const unsigned input_batch_stride = INPUT_WIDTH * INPUT_HEIGHT * NUM_INPUT;
    const unsigned output_batch_stride = OUTPUT_WIDTH * OUTPUT_HEIGHT * NUM_OUTPUT;

    const unsigned gid = get_global_id(0);

    //const unsigned gid = get_global_id(2);

    const unsigned output_fm     = gid * GROUP2;

#if GROUP2 == 16
#define realX real16
#define vloadX vload16
#endif

#if GROUP2 == 8
#define realX real8
#define vloadX vload8
#endif

#if GROUP2 == 4
#define realX real4
#define vloadX vload4
#endif

#if GROUP2 == 2
#define realX real2
#define vloadX vload2
#endif

#if GROUP2 == 1
#define realX real
#define vloadX(x,y) (*(y))
#endif

    //Do four outputs vectorized across feature maps
    realX outputs    = 0;
#if BATCH_NUM >= 4
    realX outputs1 = 0;
    realX outputs2 = 0;
    realX outputs3 = 0;
#endif
#if BATCH_NUM >= 8
    realX outputs4 = 0;
    realX outputs5 = 0;
    realX outputs6 = 0;
    realX outputs7 = 0;
#endif
#if BATCH_NUM == 16
    realX outputs8 = 0;
    realX outputs9 = 0;
    realX outputsA = 0;
    realX outputsB = 0;
    realX outputsC = 0;
    realX outputsD = 0;
    realX outputsE = 0;
    realX outputsF = 0;
#endif

    real8 inputs;
#if BATCH_NUM >= 4
    real8 inputs1;
    real8 inputs2;
    real8 inputs3;
#endif
#if BATCH_NUM >= 8
    real8 inputs4;
    real8 inputs5;
    real8 inputs6;
    real8 inputs7;
#endif
#if BATCH_NUM == 16
    real8 inputs8;
    real8 inputs9;
    real8 inputsA;
    real8 inputsB;
    real8 inputsC;
    real8 inputsD;
    real8 inputsE;
    real8 inputsF;
#endif
    realX my_weights;

    unsigned weight_idx = 36 * gid;
    const unsigned weight_stride = 36 * NUM_OUTPUT / GROUP2;            // "/ 4" is for WEIGHT_INTERLIEVE=4

    __global real* my_input  = input;
    //real8 master_inputs;
    for (unsigned input_fm = 0; input_fm < NUM_INPUT/2; ++input_fm)
    {

        for(unsigned idx=0;idx<6; idx++) // tbd idx
        {
            //master_inputs = vload8(0, my_input + INPUT_WIDTH*idx + input_batch_stride * get_sub_group_local_id());
            //inputs  = intel_sub_group_shuffle(master_inputs, 0);
            //inputs1   = intel_sub_group_shuffle(master_inputs, 1);
            //inputs2   = intel_sub_group_shuffle(master_inputs, 2);
            //inputs3   = intel_sub_group_shuffle(master_inputs, 3);
            //inputs4   = intel_sub_group_shuffle(master_inputs, 4);
            //inputs5   = intel_sub_group_shuffle(master_inputs, 5);
            //inputs6   = intel_sub_group_shuffle(master_inputs, 6);
            //inputs7   = intel_sub_group_shuffle(master_inputs, 7);

            inputs    = vload8(0, my_input + INPUT_WIDTH*idx );
        #if BATCH_NUM >= 4
            inputs1    = vload8(0, my_input + INPUT_WIDTH*idx + input_batch_stride);
            inputs2    = vload8(0, my_input + INPUT_WIDTH*idx + 2*input_batch_stride);
            inputs3    = vload8(0, my_input + INPUT_WIDTH*idx + 3*input_batch_stride);
        #endif
        #if BATCH_NUM >= 8
            inputs4    = vload8(0, my_input + INPUT_WIDTH*idx + 4*input_batch_stride);
            inputs5    = vload8(0, my_input + INPUT_WIDTH*idx + 5*input_batch_stride);
            inputs6    = vload8(0, my_input + INPUT_WIDTH*idx + 6*input_batch_stride);
            inputs7    = vload8(0, my_input + INPUT_WIDTH*idx + 7*input_batch_stride);
        #endif
        #if BATCH_NUM == 16
            inputs8    = vload8(0, my_input + INPUT_WIDTH*idx + 8  *input_batch_stride);
            inputs9    = vload8(0, my_input + INPUT_WIDTH*idx + 9  *input_batch_stride);
            inputsA    = vload8(0, my_input + INPUT_WIDTH*idx + 0xA*input_batch_stride);
            inputsB    = vload8(0, my_input + INPUT_WIDTH*idx + 0xB*input_batch_stride);
            inputsC    = vload8(0, my_input + INPUT_WIDTH*idx + 0xC*input_batch_stride);
            inputsD    = vload8(0, my_input + INPUT_WIDTH*idx + 0xD*input_batch_stride);
            inputsE    = vload8(0, my_input + INPUT_WIDTH*idx + 0xE*input_batch_stride);
            inputsF    = vload8(0, my_input + INPUT_WIDTH*idx + 0xF*input_batch_stride);
        #endif
            for(unsigned weight_c=0; weight_c<6; weight_c++)
            {
                    my_weights  = vloadX(weight_idx + 6*idx + weight_c, weights0);
                    outputs    += inputs[weight_c] * my_weights;
            #if BATCH_NUM >= 4
                    outputs1   += inputs1[weight_c] * my_weights;
                    outputs2   += inputs2[weight_c] * my_weights;
                    outputs3   += inputs3[weight_c] * my_weights;
            #endif
            #if BATCH_NUM >= 8
                    outputs4   += inputs4[weight_c] * my_weights;
                    outputs5   += inputs5[weight_c] * my_weights;
                    outputs6   += inputs6[weight_c] * my_weights;
                    outputs7   += inputs7[weight_c] * my_weights;
            #endif
            #if BATCH_NUM == 16
                    outputs8   += inputs8[weight_c] * my_weights;
                    outputs9   += inputs9[weight_c] * my_weights;
                    outputsA   += inputsA[weight_c] * my_weights;
                    outputsB   += inputsB[weight_c] * my_weights;
                    outputsC   += inputsC[weight_c] * my_weights;
                    outputsD   += inputsD[weight_c] * my_weights;
                    outputsE   += inputsE[weight_c] * my_weights;
                    outputsF   += inputsF[weight_c] * my_weights;
            #endif
            }
        }

        my_input   += INPUT_WIDTH * INPUT_HEIGHT;
        weight_idx += weight_stride;
    }

    weight_idx = 36 * gid;
    for (unsigned input_fm =NUM_INPUT/2; input_fm < NUM_INPUT; ++input_fm)
    {
        for(unsigned idx=0;idx<6; idx++) // tbd idx
        {
    inputs    = vload8(0, my_input + INPUT_WIDTH*idx );
        #if BATCH_NUM >= 4
            inputs1    = vload8(0, my_input + INPUT_WIDTH*idx + input_batch_stride);
            inputs2    = vload8(0, my_input + INPUT_WIDTH*idx + 2*input_batch_stride);
            inputs3    = vload8(0, my_input + INPUT_WIDTH*idx + 3*input_batch_stride);
        #endif
        #if BATCH_NUM >= 8
            inputs4    = vload8(0, my_input + INPUT_WIDTH*idx + 4*input_batch_stride);
            inputs5    = vload8(0, my_input + INPUT_WIDTH*idx + 5*input_batch_stride);
            inputs6    = vload8(0, my_input + INPUT_WIDTH*idx + 6*input_batch_stride);
            inputs7    = vload8(0, my_input + INPUT_WIDTH*idx + 7*input_batch_stride);
        #endif
        #if BATCH_NUM == 16
            inputs8    = vload8(0, my_input + INPUT_WIDTH*idx + 8  *input_batch_stride);
            inputs9    = vload8(0, my_input + INPUT_WIDTH*idx + 9  *input_batch_stride);
            inputsA    = vload8(0, my_input + INPUT_WIDTH*idx + 0xA*input_batch_stride);
            inputsB    = vload8(0, my_input + INPUT_WIDTH*idx + 0xB*input_batch_stride);
            inputsC    = vload8(0, my_input + INPUT_WIDTH*idx + 0xC*input_batch_stride);
            inputsD    = vload8(0, my_input + INPUT_WIDTH*idx + 0xD*input_batch_stride);
            inputsE    = vload8(0, my_input + INPUT_WIDTH*idx + 0xE*input_batch_stride);
            inputsF    = vload8(0, my_input + INPUT_WIDTH*idx + 0xF*input_batch_stride);
        #endif
            for(unsigned weight_c=0; weight_c<6; weight_c++)
            {
                    my_weights  = vloadX(weight_idx + 6*idx + weight_c, weights1);
                    outputs    += inputs[weight_c] * my_weights;
            #if BATCH_NUM >= 4
                    outputs1   += inputs1[weight_c] * my_weights;
                    outputs2   += inputs2[weight_c] * my_weights;
                    outputs3   += inputs3[weight_c] * my_weights;
            #endif
            #if BATCH_NUM >= 8
                    outputs4   += inputs4[weight_c] * my_weights;
                    outputs5   += inputs5[weight_c] * my_weights;
                    outputs6   += inputs6[weight_c] * my_weights;
                    outputs7   += inputs7[weight_c] * my_weights;
            #endif
            #if BATCH_NUM == 16
                    outputs8   += inputs8[weight_c] * my_weights;
                    outputs9   += inputs9[weight_c] * my_weights;
                    outputsA   += inputsA[weight_c] * my_weights;
                    outputsB   += inputsB[weight_c] * my_weights;
                    outputsC   += inputsC[weight_c] * my_weights;
                    outputsD   += inputsD[weight_c] * my_weights;
                    outputsE   += inputsE[weight_c] * my_weights;
                    outputsF   += inputsF[weight_c] * my_weights;
            #endif
            }
        }
        my_input   += INPUT_WIDTH * INPUT_HEIGHT;
        weight_idx += weight_stride;
    }

    unsigned out_idx;
#if GROUP2 >=2
    for(unsigned i=0; i<GROUP2; i++)
    {
        out_idx =  batch_output_offset + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs[i]);

#if BATCH_NUM >= 4
        out_idx =  batch_output_offset + output_batch_stride + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs1[i]);

        out_idx =  batch_output_offset + output_batch_stride * 2 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs2[i]);

        out_idx =  batch_output_offset + output_batch_stride * 3 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs3[i]);
#endif //BATCH_NUM >= 4
#if BATCH_NUM >= 8
        out_idx =  batch_output_offset + output_batch_stride * 4 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs4[i]);

        out_idx =  batch_output_offset + output_batch_stride * 5 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs5[i]);

        out_idx =  batch_output_offset + output_batch_stride * 6 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs6[i]);

        out_idx =  batch_output_offset + output_batch_stride * 7 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs7[i]);
#endif //BATCH_NUM >= 8
#if BATCH_NUM ==16
        out_idx =  batch_output_offset + output_batch_stride * 8 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs8[i]);

        out_idx =  batch_output_offset + output_batch_stride * 9 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs9[i]);

        out_idx =  batch_output_offset + output_batch_stride * 10 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsA[i]);

        out_idx =  batch_output_offset + output_batch_stride * 11 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsB[i]);

        out_idx =  batch_output_offset + output_batch_stride * 12 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsC[i]);

        out_idx =  batch_output_offset + output_batch_stride * 13 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsD[i]);

        out_idx =  batch_output_offset + output_batch_stride * 14 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsE[i]);

        out_idx =  batch_output_offset + output_batch_stride * 15 + (output_fm + i) * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsF[i]);
#endif //BATCH_NUM >= 16
    }

#else // GROUP2 == 1 so we don't use vectors for outputs
        out_idx =  batch_output_offset + output_fm * output_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs);

#if BATCH_NUM >= 4
        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs1);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs2);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs3);
#endif //BATCH_NUM >= 4
#if BATCH_NUM >= 8
        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs4);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs5);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs6);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs7);
#endif //BATCH_NUM >= 8
#if BATCH_NUM ==16
        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs8);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputs9);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsA);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsB);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsC);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsD);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsE);

        out_idx +=  output_batch_stride;
        output[out_idx] = activation_function(output[out_idx] + outputsF);
#endif //#if BATCH_NUM >=16


#endif  // #if GROUP2 >=2

}
#endif //INCLUDE_convolve_6x6x1_v4x1_i_readInColumns_batch_loop
)";



std::string conv_kernel7 = R"(
// macros for loop unrolling
#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define LOOP1(VAR, STMT) (STMT); (VAR)++;
#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;
#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;
#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;
#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;
#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;
#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;
#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;
#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;
#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;
#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;
#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;
#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;
#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))


#ifdef INCLUDE_convolve_AlexNet_C1

// convolution kernel width and height, must be square kernel
#define KERNEL 11
// convolution stride x an y
#define K_STRIDE 4

// each work-item iterates this many times in the width dimension
#define OUT_BLOCK_WIDTH 4
// each work-itme iterates this many times in the height dimension
#define OUT_BLOCK_HEIGHT 3

// need KERNEL width for first output + STRIDE more for each additional.
#define IN_BLOCK_WIDTH  KERNEL + K_STRIDE * (OUT_BLOCK_WIDTH  - 1)
#define IN_BLOCK_HEIGHT KERNEL + K_STRIDE * (OUT_BLOCK_HEIGHT - 1)

// Each work-item computes a 4x3 region of one output map.
// Each work-group (which will be mapped to 1 SIMD16 EU thread) will compute 16 different feature maps, but each feature map is for the same 4x3 region of the imput image.
#define SIMD_SIZE 16
// NOTE: this reqd_work_group_size does not guarantee that SIMD16 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
kernel void
convolve_AlexNet_C1 (
    __global float* outputs,
    __global float* inputs,
    filter_qualifier float* weights,
    __global float* biases,
    unsigned int batch_output_offset)  //__global float *inputs, __global float* weights, __global float* outputs)
{
    uint oc  = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column
    uint or  = get_global_id(1) * OUT_BLOCK_HEIGHT; // or = Output Row
    uint fm  = get_global_id(2);                    // fm = Feature Map = od = Output Depth
    uint fmg = get_group_id(2);
    uint lid = get_local_id(2);
    //float w;
    // 19 x 24 = 456; 456 / 16 = 29 floats per lane for SIMD16, but padding out to 30 to simplify the load loop.
    float in[30];   // this holds a 19x24 block of the input data, enough to compute 4x3 outputs, simd_shuffle is used so that all work-items have access to all 19x24 locations.
    float out[12]; // 4x3 block of outputs that is SIMD_SIZE deep (along the Feature Map dimension).
    for(int i=0;i<12;i++)
    {
        // we need this address calculation for biases because we support views and batching
        out[i] = biases[(fm - get_global_offset(2)) % _OD];
    }

    uint in_addr;
    uint weight_addr = (fmg % (_OD/SIMD_SIZE)) * _ID * KERNEL * KERNEL * SIMD_SIZE + lid;

    uint input_batch_offset = (fm / _OD) * (_IH + IHPAD) * (_IW + IWPAD) * _ID;

    for(int kd = 0; kd < _ID; kd++)  // _ID = 3, RGB
    {
        in_addr = input_batch_offset + (kd + INPUT_START_Z) * (_IH + IHPAD) * (_IW + IWPAD) + (or*K_STRIDE + INPUT_START_Y) * (_IW + IWPAD) + (oc*K_STRIDE + INPUT_START_X)  + lid;

        // read 24x19 block into registers.
        // This is ugly, we really need to fix the programming model.
        for(uint reg = 0; reg < 30; reg+=3) {
            in[reg] = inputs[in_addr];// read 16 elements
            // might be better to adjust the addrs, then do single load.
            if(lid < 8) in[reg + 1] = inputs[in_addr + 16];// read 8 elements in lower portion, for total of 24 from input row.
            in_addr += (_IW + IWPAD);  // move to next row down
            if(lid >= 8) in[reg + 1] = inputs[in_addr - 8];  // read 8 elements into upper portion
            in[reg + 2] = inputs[in_addr + 8]; // read 16 elements
            in_addr += (_IW + IWPAD);  // move to next row down
        }

        float w[5];
        int w_idx=0;
        w[0] = weights[weight_addr]; weight_addr += SIMD_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
        w[1] = weights[weight_addr]; weight_addr += SIMD_SIZE;
        w[2] = weights[weight_addr]; weight_addr += SIMD_SIZE;
        w[3] = weights[weight_addr]; weight_addr += SIMD_SIZE;
        w[4] = weights[weight_addr]; weight_addr += SIMD_SIZE;

        int kr = 0; // kr = Kernel Row
        LOOP(10, kr,  // LOOP is a macro that unrolls the loop.
        {
            int kc = 0; // kc = Kernel Column
            LOOP(KERNEL, kc,
            {
                for(int br=0; br<OUT_BLOCK_HEIGHT; br++) {
                    for(int bc=0; bc<OUT_BLOCK_WIDTH; bc++) {
                        //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * K_STRIDE + kr][bc * K_STRIDE + kc];
                        float val = intel_sub_group_shuffle( in[(((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) / SIMD_SIZE], (((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) & (SIMD_SIZE - 1));
                        out[br * OUT_BLOCK_WIDTH + bc] = mad(w[w_idx%5], val, out[br * OUT_BLOCK_WIDTH + bc]);
                    }
                }

                w[w_idx%5] = weights[weight_addr]; weight_addr += SIMD_SIZE;
                w_idx++;

            });
        });

        // last kr loop split in two parts
        int kc = 0; // kc = Kernel Column
        LOOP(6, kc,
        {
            for(int br=0; br<OUT_BLOCK_HEIGHT; br++) {
                for(int bc=0; bc<OUT_BLOCK_WIDTH; bc++) {
                    //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * K_STRIDE + kr][bc * K_STRIDE + kc];
                    float val = intel_sub_group_shuffle( in[(((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) / SIMD_SIZE], (((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) & (SIMD_SIZE - 1));
                    out[br * OUT_BLOCK_WIDTH + bc] = mad(w[w_idx%5], val, out[br * OUT_BLOCK_WIDTH + bc]);
                }
            }

            w[w_idx%5] = weights[weight_addr]; weight_addr += SIMD_SIZE;
            w_idx++;
         });

        // last 5 kc loops don't prefetch weights
        LOOP(5, kc,
        {
            for(int br=0; br<OUT_BLOCK_HEIGHT; br++) {
                for(int bc=0; bc<OUT_BLOCK_WIDTH; bc++) {
                    //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * K_STRIDE + kr][bc * K_STRIDE + kc];
                    float val = intel_sub_group_shuffle( in[(((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) / SIMD_SIZE], (((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) & (SIMD_SIZE - 1));
                    out[br * OUT_BLOCK_WIDTH + bc] = mad(w[w_idx%5], val, out[br * OUT_BLOCK_WIDTH + bc]);
                }
            }
            w_idx++;
         });

    }

    // write the 4x3 (and 16 feature maps deep) output tile to memory
    uint out_addr = OUT_BUFF_OFFSET + fm * (_OW + OWPAD) * (_OH + OHPAD); // out_addr indexes into start of 16 feature maps.
    out_addr += or * (_OW + OWPAD) + oc;  // offset for the 4x3 block that this workitem is working on;
#ifndef WRITE_PADDED_VALUES
    if(get_global_id(0) != (get_global_size(0)-1) &&
        get_global_id(1) != (get_global_size(1)-1)  )
    {
#endif
        for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
              outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
#ifndef WRITE_PADDED_VALUES
    }else if ( get_global_id(1) != (get_global_size(1)-1) )
    {
        for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < LAST_BLOCK_WIDTH; c++) {
                  outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
    }
    else if ( get_global_id(0) != (get_global_size(0)-1) )
    {
        for(uint r = 0; r < LAST_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
                  outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
    }
    else
    {
     for(uint r = 0; r < LAST_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < LAST_BLOCK_WIDTH; c++) {
                outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
    }
#endif //#ifndef WRITE_PADDED_VALUES
}

#endif  //#ifdef INCLUDE_convolve_AlexNet_C1
)";

std::string conv_kernel8 = R"(
#ifdef INCLUDE_convolve_simd16

// todo: pass these defines in to cl build.
// C2 layer
#define KERNEL FILTER_WIDTH
// convolution stride, same for x and y
#define K_STRIDE STRIDEX

#ifndef IWPAD
#define IWPAD 0
#endif

#ifndef IHPAD
#define IHPAD 0
#endif

#define OUT_BLOCK_SIZE (OUT_BLOCK_WIDTH*OUT_BLOCK_HEIGHT)

#ifndef MASTER_OUT_BLOCK_WIDTH
#define MASTER_OUT_BLOCK_WIDTH OUT_BLOCK_WIDTH
#endif
#ifndef MASTER_OUT_BLOCK_HEIGHT
#define MASTER_OUT_BLOCK_HEIGHT OUT_BLOCK_HEIGHT
#endif
// Each work-item computes a 4x6 region of one output map.
// Each work-group (which will be mapped to 1 SIMD16 EU thread) will compute 16 different feature maps, but each feature map is for the same 4x6 region of the imput image.
// NDRange:  (_OW+pad)/ OUT_BLOCK_WIDTH, (_OH+pad)/OUT_BLOCK_HEIGHT, _OD/OUT_BLOCK_DEPTH

//#define SIMD_SIZE 16
// NOTE: this reqd_work_group_size does not guarantee that SIMD16 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
kernel void
convolve_simd16(  // __global float *inputs, __global float* weights, __global float* outputs
#ifdef IMAGE_AS_OUTPUT
    __write_only image2d_t outputs,
#else
    __global float* outputs,
#endif
    __global float* inputs,
    filter_qualifier float* weights,
    __global float* biases,
    unsigned int batch_output_offset)
{
    uint oc  = get_global_id(0) * MASTER_OUT_BLOCK_WIDTH;  // oc = Output Column
    uint or  = get_global_id(1) * MASTER_OUT_BLOCK_HEIGHT; // or = Output Row
    uint fm  = get_global_id(2);                    // fm = Feature Map = od = Output Depth
    uint fmg = get_group_id(2);
    uint lid = get_local_id(2);

    float in[IN_BUFFER_SIZE];   // load 11x16 block of input data, really only need 11x15 for 4x6 outputs, but keep it simple.
    //float out[24]; // 4x6 block of outputs that is SIMD_SIZE deep (along the Feature Map dimension).
    float out[OUT_BLOCK_SIZE];

    uint in_addr;

    // find weights adress of given neuron (lid is index)
    uint weight_addr = (fmg % (_OD/SIMD_SIZE)) * INPUT_DEPTH * KERNEL * KERNEL * SIMD_SIZE + lid;

    for(int i=0;i<OUT_BLOCK_SIZE;i++) {
        out[i]=0.0f;
    }

    uint num_in_batch = ( fm - get_global_offset(2) ) / _OD ;

    uint input_batch_offset = num_in_batch * (_IH + IHPAD) * (_IW + IWPAD) * TOTAL_INPUT_DEPTH_SIZE;
    for(int kd = 0; kd < _ID; kd++)
    {
        in_addr = input_batch_offset + (kd + INPUT_START_Z) * (_IH + IHPAD) * (_IW + IWPAD) + (or*K_STRIDE + INPUT_START_Y) * (_IW + IWPAD) + (oc*K_STRIDE + INPUT_START_X)  + lid;

        // read 11x16 input block into registers.
        for(uint reg = 0; reg < IN_BUFFER_SIZE; reg++) {
            in[reg] = inputs[in_addr];// read 16 elements
            in_addr += (_IW + IWPAD);  // move to next row down
        }
#define WEIGHT_PREF 5
        float w[WEIGHT_PREF];
        int w_idx=0;

        LOOP(WEIGHT_PREF, w_idx,  // LOOP is a macro that unrolls the loop.
        {
            w[w_idx] = weights[weight_addr]; weight_addr += SIMD_SIZE;
        });

        int kr = 0; // kr = Kernel Row
        LOOP(KERNEL, kr,  // LOOP is a macro that unrolls the loop.
        {
            int kc = 0; // kc = Kernel Column
            LOOP(KERNEL, kc,
            {
                for(int br=0; br < OUT_BLOCK_HEIGHT; br++) {
                    for(int bc=0; bc < OUT_BLOCK_WIDTH; bc++) {
                        float input = intel_sub_group_shuffle( in[br * K_STRIDE + kr], bc * K_STRIDE + kc);
                        out[br * OUT_BLOCK_WIDTH + bc] = mad(w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);
                    }
                }
                w[w_idx % WEIGHT_PREF] = weights[weight_addr];
                weight_addr += SIMD_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                ++w_idx;
            });
        });

        // We advanced weight_addr too far in last 5 loop iterations
        weight_addr -= WEIGHT_PREF * SIMD_SIZE;
    }

#ifdef IMAGE_AS_OUTPUT
    // TODO: no ULT for that one yet!
    uint out_addr = ( num_in_batch * TOTAL_OUTPUT_DEPTH  + (fm % _OD) + get_global_offset(2) ) * (_OW + OWPAD) * (_OH + OHPAD); // out_addr indexes into start of 16 feature maps.
#else
    // we need this address calculation for outputs because we support views and batching
    uint out_addr = OUT_BUFF_OFFSET + ( num_in_batch * TOTAL_OUTPUT_DEPTH  + (fm % _OD) + get_global_offset(2) ) * (_OW + OWPAD) * (_OH + OHPAD);
#endif

    out_addr += or * (_OW + OWPAD) + oc;  // offset for the 4x3 block that this workitem is working on;

    // we need this address calculation for biases because we support views and batching
    float bias = biases[(fm - get_global_offset(2)) % _OD ];
#ifndef WRITE_PADDED_VALUES
    if(get_global_id(0) != (get_global_size(0)-1) &&
        get_global_id(1) != (get_global_size(1)-1)  )
    {
#endif
        for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
#ifdef IMAGE_AS_OUTPUT
                write_imagef(outputs,(int2)(out_addr + r * (_OW + OWPAD) + c,num_in_batch),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
                outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
            }
        }
#ifndef WRITE_PADDED_VALUES
    }else if ( get_global_id(1) != (get_global_size(1)-1) )
    {
        for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < LAST_BLOCK_WIDTH; c++) {
#ifdef IMAGE_AS_OUTPUT
                    write_imagef(outputs,(int2)(out_addr + r * (_OW + OWPAD) + c,num_in_batch),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
                    outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
            }
        }
    }
    else if ( get_global_id(0) != (get_global_size(0)-1) )
    {
        for(uint r = 0; r < LAST_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
#ifdef IMAGE_AS_OUTPUT
               write_imagef(outputs,(int2)(out_addr + r * (_OW + OWPAD) + c,num_in_batch),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
               outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
            }
        }
    }
    else
    {
     for(uint r = 0; r < LAST_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < LAST_BLOCK_WIDTH; c++) {
#ifdef IMAGE_AS_OUTPUT
                    write_imagef(outputs,(int2)(c,r*(_OW + OWPAD)),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
                    outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
            }
        }
    }
#endif //#ifndef WRITE_PADDED_VALUES
}

#endif  //INCLUDE_convolve_simd16
)";

std::string conv_kernel9 = R"(
#ifdef INCLUDE_convolve_simd8

// todo: pass these defines in to cl build.
// C2 layer
#define KERNEL FILTER_WIDTH
// convolution stride, same for x and y
#define K_STRIDE STRIDEX

#ifndef IWPAD
#define IWPAD 0
#endif

#ifndef IHPAD
#define IHPAD 0
#endif

#define OUT_BLOCK_SIZE (OUT_BLOCK_WIDTH*OUT_BLOCK_HEIGHT)

#ifndef MASTER_OUT_BLOCK_WIDTH
#define MASTER_OUT_BLOCK_WIDTH OUT_BLOCK_WIDTH
#endif
#ifndef MASTER_OUT_BLOCK_HEIGHT
#define MASTER_OUT_BLOCK_HEIGHT OUT_BLOCK_HEIGHT
#endif

// Each work-item computes a 4x6 region of one output map.
// Each work-group (which will be mapped to 1 SIMD16 EU thread) will compute 16 different feature maps, but each feature map is for the same 4x6 region of the imput image.
// NDRange:  (_OW+pad)/ OUT_BLOCK_WIDTH, (_OH+pad)/OUT_BLOCK_HEIGHT, _OD/OUT_BLOCK_DEPTH

// NOTE: this reqd_work_group_size does not guarantee that SIMD16 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
kernel void
convolve_simd8(  // __global float *inputs, __global float* weights, __global float* outputs
    __global float* outputs,
    __global float* inputs,
    filter_qualifier float* weights,
    __global float* biases,
    unsigned int batch_output_offset)
{
    uint oc  = get_global_id(0) * MASTER_OUT_BLOCK_WIDTH;  // oc = Output Column
    uint or  = get_global_id(1) * MASTER_OUT_BLOCK_HEIGHT; // or = Output Row
    uint fm  = get_global_id(2);                    // fm = Feature Map = od = Output Depth
    uint fmg = get_group_id(2);
    uint lid = get_local_id(2);


    float in[IN_BUFFER_SIZE];   // load 11x16 block of input data, really only need 11x15 for 4x6 outputs, but keep it simple.
    //float out[24]; // 4x6 block of outputs that is SIMD_SIZE deep (along the Feature Map dimension).
    float out[OUT_BLOCK_SIZE];

    uint in_addr;
    uint weight_addr = (fmg % (_OD/SIMD_SIZE)) * _ID * KERNEL * KERNEL * SIMD_SIZE + lid;

    for(int i=0;i<OUT_BLOCK_SIZE;i++)
    {
        // we need this address calculation for biases because we support views and batching
        out[i] = biases[(fm - get_global_offset(2)) % _OD];
    }

    uint num_in_batch = ( fm - get_global_offset(2) ) / _OD ;
    uint input_batch_offset = num_in_batch * (_IH + IHPAD) * (_IW + IWPAD) * TOTAL_INPUT_DEPTH_SIZE;
    for(int kd = 0; kd < _ID; kd++) // 48
    {
        in_addr = input_batch_offset + (kd + INPUT_START_Z) * (_IH + IHPAD) * (_IW + IWPAD) + (or * K_STRIDE) * (_IW + IWPAD) + (oc * K_STRIDE) + lid;

        // read 11x16 input block into registers.
        for(uint reg = 0; reg < IN_BUFFER_SIZE; reg++) {
            in[reg] = inputs[in_addr];// read 16 elements
            in_addr += (_IW + IWPAD);  // move to next row down
        }

#define WEIGHT_PREF 5
        float w[WEIGHT_PREF];

        int w_idx=0;

        LOOP(WEIGHT_PREF, w_idx,  // LOOP is a macro that unrolls the loop.
        {
            w[w_idx] = weights[weight_addr]; weight_addr += SIMD_SIZE;
        });

        int kr = 0; // kr = Kernel Row
        LOOP(KERNEL, kr,  // LOOP is a macro that unrolls the loop.
        {
            int kc = 0; // kc = Kernel Column
            LOOP(2, kc,
            {
                for(int br=0; br < OUT_BLOCK_HEIGHT; br++) {
                    for(int bc=0; bc < OUT_BLOCK_WIDTH; bc++) {
                        float input = intel_sub_group_shuffle( in[br * K_STRIDE + kr], bc * K_STRIDE + kc);
                        out[br * OUT_BLOCK_WIDTH + bc] = mad(w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);
                    }
                }
                w[w_idx % WEIGHT_PREF] = weights[weight_addr];
                weight_addr += SIMD_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                ++w_idx;
            });

            for(int br=0; br < OUT_BLOCK_HEIGHT; br++) {
                    for(int bc=0; bc < OUT_BLOCK_WIDTH-1; bc++) {
                        float input = intel_sub_group_shuffle( in[br * K_STRIDE + kr], bc * K_STRIDE + kc);
                        out[br * OUT_BLOCK_WIDTH + bc] = mad(w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);
                    }
                }

             // simd8, so 9th input read  separately
#if OUT_BLOCK_WIDTH==7
             in_addr = input_batch_offset + (kd + INPUT_START_Z)  * (_IH + IHPAD) * (_IW + IWPAD) + (or * K_STRIDE + kr) * (_IW + IWPAD) + (oc * K_STRIDE) + 8;
#endif
             for(int br=0; br < OUT_BLOCK_HEIGHT; br++) {
                   const int bc = OUT_BLOCK_WIDTH-1;
#if OUT_BLOCK_WIDTH==7
                   float input = inputs[in_addr];
#else
                   float input = intel_sub_group_shuffle( in[br * K_STRIDE + kr], bc * K_STRIDE + kc);
#endif
                   in_addr += (_IW + IWPAD);  // move to next row down
                   out[br * OUT_BLOCK_WIDTH + bc] = mad(w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);
                }

            w[w_idx % WEIGHT_PREF] = weights[weight_addr];
            weight_addr += SIMD_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
            ++w_idx;

        });

        // We advanced weight_addr too far in last 5 loop iterations
        weight_addr -= WEIGHT_PREF * SIMD_SIZE;
    }


    // we need this address calculation for outputs because we support views and batching
    uint out_addr = OUT_BUFF_OFFSET + ( num_in_batch * TOTAL_OUTPUT_DEPTH  + (fm % _OD) + get_global_offset(2) ) * (_OW + OWPAD) * (_OH + OHPAD);

    out_addr += or * (_OW + OWPAD) + oc;  // offset for the 4x3 block that this workitem is working on;

#ifndef WRITE_PADDED_VALUES
    if(get_global_id(0) != (get_global_size(0)-1) &&
        get_global_id(1) != (get_global_size(1)-1)  )
    {
#endif
        for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
              outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
#ifndef WRITE_PADDED_VALUES
    }else if ( get_global_id(1) != (get_global_size(1)-1) )
    {
        for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < LAST_BLOCK_WIDTH; c++) {
                  outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
    }
    else if ( get_global_id(0) != (get_global_size(0)-1) )
    {
        for(uint r = 0; r < LAST_BLOCK_HEIGHT; r++) {
            for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
                  outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
    }
    else
    {
     for(uint r = 0; r < LAST_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < LAST_BLOCK_WIDTH; c++) {
                outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(out[r * OUT_BLOCK_WIDTH + c]);
            }
        }
    }
#endif //#ifndef WRITE_PADDED_VALUES
}

#endif  //INCLUDE_convolve_simd8
)";


std::string conv_kernel10 = R"(
#ifdef INCLUDE_convolve_AlexNet_C1_original

// convolution kernel width and height, must be square kernel
#define KERNEL 11
// convolution stride x an y
#define K_STRIDE 4

// each work-item iterates this many times in the width dimension
#define OUT_BLOCK_WIDTH 4
// each work-itme iterates this many times in the height dimension
#define OUT_BLOCK_HEIGHT 3

// need KERNEL width for first output + STRIDE more for each additional.
#define IN_BLOCK_WIDTH  KERNEL + K_STRIDE * (OUT_BLOCK_WIDTH  - 1)
#define IN_BLOCK_HEIGHT KERNEL + K_STRIDE * (OUT_BLOCK_HEIGHT - 1)

// Each work-item computes a 4x3 region of one output map.
// Each work-group (which will be mapped to 1 SIMD16 EU thread) will compute 16 different feature maps, but each feature map is for the same 4x3 region of the imput image.
#define SIMD_SIZE 16
// NOTE: this reqd_work_group_size does not guarantee that SIMD16 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
kernel void
convolve_AlexNet_C1_original (
    __global float* outputs,
    __global float* inputs,
    filter_qualifier float* weights,
    unsigned int batch_output_offset)  //__global float *inputs, __global float* weights, __global float* outputs)
{
    uint oc  = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column
    uint or  = get_global_id(1) * OUT_BLOCK_HEIGHT; // or = Output Row
    uint fm  = get_global_id(2);                    // fm = Feature Map = od = Output Depth
    uint fmg = get_group_id(2);
    uint lid = get_local_id(2);
    float w;
    // 19 x 24 = 456; 456 / 16 = 29 floats per lane for SIMD16, but padding out to 30 to simplify the load loop.
    float in[30];   // this holds a 19x24 block of the input data, enough to compute 4x3 outputs, simd_shuffle is used so that all work-items have access to all 19x24 locations.
    float out[12]; // 4x3 block of outputs that is SIMD_SIZE deep (along the Feature Map dimension).
    for(int i=0;i<12;i++) { out[i]=0.0f;}  // todo: init these with the neural net biases.

    uint in_addr;
    uint weight_addr = fmg * _ID * KERNEL * KERNEL * SIMD_SIZE + lid;

    for(int kd = 0; kd < _ID; kd++)  // _ID = 3, RGB
    {

        in_addr = kd * (_IH + IHPAD) * (_IW + IWPAD) + (or * K_STRIDE) * (_IW + IWPAD) + (oc * K_STRIDE) + lid;

        // read 24x19 block into registers.
        // This is ugly, we really need to fix the programming model.
        for(uint reg = 0; reg < 30; reg+=3) {
            in[reg] = inputs[in_addr];// read 16 elements
            // might be better to adjust the addrs, then do single load.
            if(lid < 8) in[reg + 1] = inputs[in_addr + 16];// read 8 elements in lower portion, for total of 24 from input row.
            in_addr += (_IW + IWPAD);  // move to next row down
            if(lid >= 8) in[reg + 1] = inputs[in_addr - 8];  // read 8 elements into upper portion
            in[reg + 2] = inputs[in_addr + 8]; // read 16 elements
            in_addr += (_IW + IWPAD);  // move to next row down
        }

        int kr = 0; // kr = Kernel Row
        LOOP(KERNEL, kr,  // LOOP is a macro that unrolls the loop.
        {
            int kc = 0; // kc = Kernel Column
            LOOP(KERNEL, kc,
            {
                w = weights[weight_addr];
                for(int br=0; br<OUT_BLOCK_HEIGHT; br++) {
                    for(int bc=0; bc<OUT_BLOCK_WIDTH; bc++) {
                        //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * K_STRIDE + kr][bc * K_STRIDE + kc];
                        float val = intel_sub_group_shuffle( in[(((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) / SIMD_SIZE], (((br*K_STRIDE+kr)*24)+(bc * K_STRIDE + kc)) & (SIMD_SIZE - 1));
                        out[br * OUT_BLOCK_WIDTH + bc] = mad(w, val, out[br * OUT_BLOCK_WIDTH + bc]);
                    }
                }
                weight_addr += SIMD_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
            });
        });
    }


    // write the 4x3 (and 16 feature maps deep) output tile to memory
    uint out_addr = batch_output_offset + fm * (_OW + OWPAD) * (_OH + OHPAD); // out_addr indexes into start of 16 feature maps.
    out_addr += or * (_OW + OWPAD) + oc;  // offset for the 4x3 block that this workitem is working on;

    for(uint r = 0; r < OUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUT_BLOCK_WIDTH; c++) {
            // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
            outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(outputs[out_addr + r * (_OW + OWPAD) + c] + out[r * OUT_BLOCK_WIDTH + c]);
        }
    }
}

#endif  //#ifdef INCLUDE_convolve_AlexNet_C1_original
)";
