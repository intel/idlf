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


#include "tester/common/workflows_for_tests.h"

enum  workflow_layers {
    input,
    mean_substract,
    convert,
    conv1,
    pool1,
    norm1,
    subv1_1,
    subv1_2,
    conv2_1,
    conv2_2,
    merge2,
    pool2,
    norm2,
    conv3,
    subv3_1,
    subv3_2,
    conv4_1,
    conv4_2,
    conv5_1,
    conv5_2,
    merge5,
    pool5,
    fc6,
    fc7,
    fc8,
    softmax,
    output,
    last_workflow_item = output
};

enum  workflow_layer_factor {
    conv1_factor,
    conv2_1_factor,
    conv2_2_factor,
    conv3_factor,
    conv4_1_factor,
    conv4_2_factor,
    conv5_1_factor,
    conv5_2_factor,
    fc6_factor,
    fc7_factor,
    fc8_factor,
    last_factor = fc8_factor
};


class workflow_for_test_caffenet_int16_learned: public workflows_for_tests_base
{

public:
    workflow_for_test_caffenet_int16_learned() {
        for(auto wl : workflow_layer) wl = nullptr;
        for(auto wlwf : workflow_layer_weights_float) wlwf = nullptr;
        for(auto wlbf : workflow_layer_weights_float) wlbf = nullptr;
        for(auto wlwi : workflow_layer_weights_int16) wlwi = nullptr;
        for(auto wlbi : workflow_layer_biases_int32) wlbi = nullptr;
    }

    bool is_valid() { return error_.empty(); }

private:
    std::string error_;

    uint16_t img_size = 227;

    // pointers to successive workflow parts
    nn_workflow_item_t        *workflow_layer[last_workflow_item+1];

    // pointers to nn_datas containing weights and biases;
    nn::data<float>           *workflow_layer_weights_float[last_factor+1];
    nn::data<float>           *workflow_layer_biases_float[last_factor+1];
    nn::data<int16_t>         *workflow_layer_weights_int16[last_factor+1];
    nn::data<int32_t>         *workflow_layer_biases_int32[last_factor+1];
    nn::data<float>           *mean_factor = nullptr;

    nn_workflow_t             *workflow = nullptr;
    nn_device_interface_0_t   *di = nullptr;

public:

    virtual nn_workflow_t *init_test_workflow(nn_device_interface_0_t *_di) {

        if(!is_valid()) throw std::runtime_error(error_);

        this->di = _di;

            // load nn:data factors (weights and biases) for successive layers
            mean_factor                                    = nn_data_load_from_file("weights_caffenet/imagenet_mean.nnd");
            workflow_layer_weights_float[conv1_factor]     = nn_data_load_from_file("weights_caffenet/conv1_weights.nnd");
            workflow_layer_biases_float[conv1_factor]      = nn_data_load_from_file("weights_caffenet/conv1_biases.nnd");
            workflow_layer_weights_float[conv2_1_factor]   = nn_data_load_from_file("weights_caffenet/conv2_g1_weights.nnd");
            workflow_layer_biases_float[conv2_1_factor]    = nn_data_load_from_file("weights_caffenet/conv2_g1_biases.nnd");
            workflow_layer_weights_float[conv2_2_factor]   = nn_data_load_from_file("weights_caffenet/conv2_g2_weights.nnd");
            workflow_layer_biases_float[conv2_2_factor]    = nn_data_load_from_file("weights_caffenet/conv2_g2_biases.nnd");
            workflow_layer_weights_float[conv3_factor]     = nn_data_load_from_file("weights_caffenet/conv3_weights.nnd");
            workflow_layer_biases_float[conv3_factor]      = nn_data_load_from_file("weights_caffenet/conv3_biases.nnd");
            workflow_layer_weights_float[conv4_1_factor]   = nn_data_load_from_file("weights_caffenet/conv4_g1_weights.nnd");
            workflow_layer_biases_float[conv4_1_factor]    = nn_data_load_from_file("weights_caffenet/conv4_g1_biases.nnd");
            workflow_layer_weights_float[conv4_2_factor]   = nn_data_load_from_file("weights_caffenet/conv4_g2_weights.nnd");
            workflow_layer_biases_float[conv4_2_factor]    = nn_data_load_from_file("weights_caffenet/conv4_g2_biases.nnd");
            workflow_layer_weights_float[conv5_1_factor]   = nn_data_load_from_file("weights_caffenet/conv5_g1_weights.nnd");
            workflow_layer_biases_float[conv5_1_factor]    = nn_data_load_from_file("weights_caffenet/conv5_g1_biases.nnd");
            workflow_layer_weights_float[conv5_2_factor]   = nn_data_load_from_file("weights_caffenet/conv5_g2_weights.nnd");
            workflow_layer_biases_float[conv5_2_factor]    = nn_data_load_from_file("weights_caffenet/conv5_g2_biases.nnd");
            workflow_layer_weights_float[fc6_factor]       = nn_data_load_from_file("weights_caffenet/fc6_weights.nnd");
            workflow_layer_biases_float[fc6_factor]        = nn_data_load_from_file("weights_caffenet/fc6_biases.nnd");
            workflow_layer_weights_float[fc7_factor]       = nn_data_load_from_file("weights_caffenet/fc7_weights.nnd");
            workflow_layer_biases_float[fc7_factor]        = nn_data_load_from_file("weights_caffenet/fc7_biases.nnd");
            workflow_layer_weights_float[fc8_factor]       = nn_data_load_from_file("weights_caffenet/fc8_weights.nnd");
            workflow_layer_biases_float[fc8_factor]        = nn_data_load_from_file("weights_caffenet/fc8_biases.nnd");

            for (auto wlwf : workflow_layer_weights_float)
               if (wlwf == nullptr)
                  throw  std::runtime_error("error: one or more of file with weights was not loaded");
            for (auto wlbf : workflow_layer_biases_float)
               if (wlbf == nullptr)
                  throw  std::runtime_error("error: one or more of file with biases was not loaded");

        di->workflow_create_function(&workflow,1,1);

        //                                                            { c1    c2_1  c2_2  c3    c4_1  c4_2  c5_1  c5_2  fc6   fc7   fc8   }
        const size_t nnwrkld_accumulator_fraction[last_factor+1]    = { 16,   19,   17,   22,   22,   22,   23,   22,   24,   26,   24    };
        const size_t nnwrkld_output_fraction[last_factor+1]         = { 3,    7,    7,    6,    7,    7,    8,    8,    10,   12,   26    };
        const size_t nnwrkld_weights_float_fraction[last_factor+1]  = { 16,   16,   14,   15,   16,   16,   16,   15,   16,   16,   12    };
        const size_t nnwrkld_biases_float_fraction[last_factor+1]   = {nnwrkld_accumulator_fraction[conv1_factor],
                                                                       nnwrkld_accumulator_fraction[conv2_1_factor],
                                                                       nnwrkld_accumulator_fraction[conv2_2_factor],
                                                                       nnwrkld_accumulator_fraction[conv3_factor],
                                                                       nnwrkld_accumulator_fraction[conv4_1_factor],
                                                                       nnwrkld_accumulator_fraction[conv4_2_factor],
                                                                       nnwrkld_accumulator_fraction[conv5_1_factor],
                                                                       nnwrkld_accumulator_fraction[conv5_2_factor],
                                                                       nnwrkld_accumulator_fraction[fc6_factor],
                                                                       nnwrkld_accumulator_fraction[fc7_factor],
                                                                       nnwrkld_accumulator_fraction[fc8_factor]
                                                                      };
        for(auto i = 0; i<=last_factor;++i) {
            workflow_layer_weights_int16[i] = new nn::data<int16_t>(static_cast<const size_t*>(workflow_layer_weights_float[i]->size),workflow_layer_weights_float[i]->dimension);
            workflow_layer_biases_int32[i] = new nn::data<int32_t>(static_cast<const size_t*>(workflow_layer_biases_float[i]->size),workflow_layer_biases_float[i]->dimension);
            nn_data_convert_float_to_int16_fixedpoint(workflow_layer_weights_float[i],workflow_layer_weights_int16[i],1 << nnwrkld_weights_float_fraction[i]);
            nn_data_convert_float_to_int32_fixedpoint(workflow_layer_biases_float[i],workflow_layer_biases_int32[i],1 << nnwrkld_biases_float_fraction[i]);
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 0 (input)
        //         output: 227x227x3
        {
            di->workflow_item_create_function(&workflow_layer[input],0,nullptr,1);

            workflow_layer[input]->type = NN_WORK_ITEM_TYPE_INPUT;
            workflow_layer[input]->arguments.input.index = 0;
            workflow_layer[input]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[input]->output_format[0].format_3d ={{img_size,img_size,3}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 0 (imagenet_mean_subtract)
        //         output: 227x227x3
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[input],0};
            di->workflow_item_create_function(&workflow_layer[mean_substract],1,&inputs_descriptor,1);

            workflow_layer[mean_substract]->type = NN_WORK_ITEM_TYPE_ARITHMETIC;
            workflow_layer[mean_substract]->arguments.forward_arithmetic.factor = mean_factor;
            workflow_layer[mean_substract]->arguments.forward_arithmetic.arithmetic_function = NN_ARITHMETIC_FUNCTION_SUBTRACTION;

            workflow_layer[mean_substract]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[mean_substract]->output_format[0].format_3d ={{img_size,img_size,3}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 0 Convert float to int16
        //
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[mean_substract], 0 };
            di->workflow_item_create_function(&workflow_layer[convert], 1, &inputs_descriptor, 1);

            workflow_layer[convert]->type = NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT;
            workflow_layer[convert]->arguments.forward_convert_float_to_int16_fixedpoint.output_fraction = 0;

            workflow_layer[convert]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[convert]->output_format[0].format_3d = nn_output_format_3d{ { img_size, img_size, 4 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 01
        //           convo: 11x11 stride 4x4; ReLU; output: 55x55x96
        //         maxpool: 3x3 stride 2x2;
        //            norm: RESPONSE_ACROSS_MAPS
        //          output: 27x27x96
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[convert], 0 };
            di->workflow_item_create_function(&workflow_layer[conv1], 1, &inputs_descriptor, 1);

            workflow_layer[conv1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv1]->name = "c1";

            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;

            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv1_factor];
            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv1_factor];

            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 0;
            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 0;

            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 4;
            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 4;

            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv1_factor];
            workflow_layer[conv1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv1_factor];

            workflow_layer[conv1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv1]->output_format[0].format_3d = { { 55, 55, 96 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[conv1], 0 };
            di->workflow_item_create_function(&workflow_layer[pool1], 1, &inputs_descriptor, 1);

            workflow_layer[pool1]->type = NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT;
            workflow_layer[pool1]->name = "p1";

            workflow_layer[pool1]->arguments.forward_pooling_fixedpoint.pool_size[0] = 3;
            workflow_layer[pool1]->arguments.forward_pooling_fixedpoint.pool_size[1] = 3;
            workflow_layer[pool1]->arguments.forward_pooling_fixedpoint.pool_stride[0] = 2;
            workflow_layer[pool1]->arguments.forward_pooling_fixedpoint.pool_stride[1] = 2;

            workflow_layer[pool1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[pool1]->output_format[0].format_3d = { { 27, 27, 96 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[pool1], 0 };
            di->workflow_item_create_function(&workflow_layer[norm1], 1, &inputs_descriptor, 1);

            workflow_layer[norm1]->type = NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN;
            workflow_layer[norm1]->name = "lrn1";

            workflow_layer[norm1]->arguments.normalization_response_across_maps_forward_i16qn.k = 1;
            workflow_layer[norm1]->arguments.normalization_response_across_maps_forward_i16qn.n = 5;
            workflow_layer[norm1]->arguments.normalization_response_across_maps_forward_i16qn.alpha = 0.00002f;
            workflow_layer[norm1]->arguments.normalization_response_across_maps_forward_i16qn.beta = 0.75f;
            workflow_layer[norm1]->arguments.normalization_response_across_maps_forward_i16qn.fractions.input = nnwrkld_output_fraction[conv1_factor];
            workflow_layer[norm1]->arguments.normalization_response_across_maps_forward_i16qn.fractions.output = nnwrkld_output_fraction[conv1_factor];

            workflow_layer[norm1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[norm1]->output_format[0].format_3d = { { 27, 27, 96 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 02
        //           split: 2 (z-axis 96/2); output 27x27x(2*96/2)
        //           convo: 5x5 stride 1x1; ReLU; 0-padded output: 27x27x(2*256/2)
        //           merge: (z-axis)
        //         maxpool: 3x3 stride 2x2;
        //            norm: RESPONSE_ACROSS_MAPS
        //          output: 13x13x256
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[norm1], 0 };
            di->workflow_item_create_function(&workflow_layer[subv1_1], 1, &inputs_descriptor, 1); // view g1

            workflow_layer[subv1_1]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv1_1]->arguments.view.origin[0] = 0;
            workflow_layer[subv1_1]->arguments.view.origin[1] = 0;
            workflow_layer[subv1_1]->arguments.view.origin[2] = 0;

            workflow_layer[subv1_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv1_1]->output_format[0].format_3d = { { 27, 27, 96 / 2 } };

        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[norm1], 0 };
            di->workflow_item_create_function(&workflow_layer[subv1_2], 1, &inputs_descriptor, 1);   // view g2

            workflow_layer[subv1_2]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv1_2]->arguments.view.origin[0] = 0;
            workflow_layer[subv1_2]->arguments.view.origin[1] = 0;
            workflow_layer[subv1_2]->arguments.view.origin[2] = (96 / 2);

            workflow_layer[subv1_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv1_2]->output_format[0].format_3d = { { 27, 27, 96 / 2 } };
        }

        // convolution 2, g1: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[subv1_1], 0 };
            di->workflow_item_create_function(&workflow_layer[conv2_1], 1, &inputs_descriptor, 1);

            workflow_layer[conv2_1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv2_1]->name = "c2g1";

            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv2_1_factor];
            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv2_1_factor];

            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 2;
            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 2;

            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 1;
            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 1;

            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv2_1_factor];
            workflow_layer[conv2_1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv2_1_factor];

            workflow_layer[conv2_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv2_1]->output_format[0].format_3d = { { 27, 27, 256 / 2 } };
        }

        // convolution 2, g2: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[subv1_2], 0 };
            di->workflow_item_create_function(&workflow_layer[conv2_2], 1, &inputs_descriptor, 1);

            workflow_layer[conv2_2]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv2_2]->name = "c2g2";

            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv2_2_factor];
            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv2_2_factor];

            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 2;
            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 2;

            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 1;
            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 1;

            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv2_2_factor];
            workflow_layer[conv2_2]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv2_2_factor];

            workflow_layer[conv2_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv2_2]->output_format[0].format_3d = { { 27, 27, 256 / 2 } };
        }

        // merge g1 and g2
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_layer[conv2_1], 0 }, { workflow_layer[conv2_2], 0 } };
            di->workflow_item_create_function(&workflow_layer[merge2], 2, inputs_descriptor, 1);

            workflow_layer[merge2]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_layer[merge2]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_layer[merge2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[merge2]->output_format[0].format_3d = { { 27, 27, 256 } };
        }

        // maxpool: 3x3 stride 2x2;
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[merge2], 0 };
            di->workflow_item_create_function(&workflow_layer[pool2], 1, &inputs_descriptor, 1); // pooling

            workflow_layer[pool2]->type = NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT;
            workflow_layer[pool2]->name = "p2";

            workflow_layer[pool2]->arguments.forward_pooling_fixedpoint.pool_size[0] = 3;
            workflow_layer[pool2]->arguments.forward_pooling_fixedpoint.pool_size[1] = 3;

            workflow_layer[pool2]->arguments.forward_pooling_fixedpoint.pool_stride[0] = 2;
            workflow_layer[pool2]->arguments.forward_pooling_fixedpoint.pool_stride[1] = 2;

            workflow_layer[pool2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[pool2]->output_format[0].format_3d = { { 13, 13, 256 } };
        }

        //norm: RESPONSE_ACROSS_MAPS; output: 13x13x256
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[pool2], 0 };
            di->workflow_item_create_function(&workflow_layer[norm2], 1, &inputs_descriptor, 1);

            workflow_layer[norm2]->type = NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN;
            workflow_layer[norm2]->name = "lrn2";

            workflow_layer[norm2]->arguments.normalization_response_across_maps_forward_i16qn.k = 1;
            workflow_layer[norm2]->arguments.normalization_response_across_maps_forward_i16qn.n = 5;
            workflow_layer[norm2]->arguments.normalization_response_across_maps_forward_i16qn.alpha = 0.00002f;
            workflow_layer[norm2]->arguments.normalization_response_across_maps_forward_i16qn.beta = 0.75f;
            workflow_layer[norm2]->arguments.normalization_response_across_maps_forward_i16qn.fractions.input = nnwrkld_output_fraction[conv2_2_factor];
            workflow_layer[norm2]->arguments.normalization_response_across_maps_forward_i16qn.fractions.output = nnwrkld_output_fraction[conv2_2_factor];

            workflow_layer[norm2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[norm2]->output_format[0].format_3d = { { 13, 13, 256 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 03
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x384
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[norm2], 0 };
            di->workflow_item_create_function(&workflow_layer[conv3], 1, &inputs_descriptor, 1);

            workflow_layer[conv3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv3]->name = "c3";

            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv3_factor];
            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv3_factor];

            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 1;
            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 1;

            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 1;
            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 1;

            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv3_factor];
            workflow_layer[conv3]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv3_factor];

            workflow_layer[conv3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv3]->output_format[0].format_3d = { { 13, 13, 384 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 04
        //           split: 2 (z-axis 384/2)
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x(2*384/2) (continue split to next stage)
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[conv3], 0 };
            di->workflow_item_create_function(&workflow_layer[subv3_1], 1, &inputs_descriptor, 1); // view g1

            workflow_layer[subv3_1]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv3_1]->arguments.view.origin[0] = 0;
            workflow_layer[subv3_1]->arguments.view.origin[1] = 0;
            workflow_layer[subv3_1]->arguments.view.origin[2] = 0;

            workflow_layer[subv3_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv3_1]->output_format[0].format_3d = { { 13, 13, 384 / 2 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[conv3], 0 };
            di->workflow_item_create_function(&workflow_layer[subv3_2], 1, &inputs_descriptor, 1); // view g2

            workflow_layer[subv3_2]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv3_2]->arguments.view.origin[0] = 0;
            workflow_layer[subv3_2]->arguments.view.origin[1] = 0;
            workflow_layer[subv3_2]->arguments.view.origin[2] = 384 / 2;

            workflow_layer[subv3_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv3_2]->output_format[0].format_3d = { { 13, 13, 384 / 2 } };

        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[subv3_1], 0 };
            di->workflow_item_create_function(&workflow_layer[conv4_1], 1, &inputs_descriptor, 1); // conv g1

            workflow_layer[conv4_1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv4_1]->name = "c4g1";

            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv4_1_factor];
            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv4_1_factor];

            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 1;
            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 1;

            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 1;
            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 1;

            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv4_1_factor];
            workflow_layer[conv4_1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv4_1_factor];

            workflow_layer[conv4_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv4_1]->output_format[0].format_3d = { { 13, 13, 384 / 2 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[subv3_2], 0 };
            di->workflow_item_create_function(&workflow_layer[conv4_2], 1, &inputs_descriptor, 1); // conv g2

            workflow_layer[conv4_2]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv4_2]->name = "c4g2";

            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv4_2_factor];
            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv4_2_factor];

            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 1;
            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 1;

            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 1;
            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 1;

            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv4_2_factor];
            workflow_layer[conv4_2]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv4_2_factor];

            workflow_layer[conv4_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv4_2]->output_format[0].format_3d = { { 13, 13, 384 / 2 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 05
        //           convo: 3x3 stride 1x1; ReLU; 0-padded; output: 13x13x(2*256/2)
        //           merge: (z-axis)
        //         maxpool: 3x3 stride 2x2;
        //          output: 13x13x256
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[conv4_1], 0 };
            di->workflow_item_create_function(&workflow_layer[conv5_1], 1, &inputs_descriptor, 1); // conv g1

            workflow_layer[conv5_1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv5_1]->name = "c5g1";

            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv5_1_factor];
            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv5_1_factor];

            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 1;
            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 1;

            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 1;
            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 1;

            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv5_1_factor];
            workflow_layer[conv5_1]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv5_1_factor];

            workflow_layer[conv5_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv5_1]->output_format[0].format_3d = { { 13, 13, 256 / 2 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[conv4_2], 0 };
            di->workflow_item_create_function(&workflow_layer[conv5_2], 1, &inputs_descriptor, 1); // conv g2

            workflow_layer[conv5_2]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
            workflow_layer[conv5_2]->name = "c5g2";

            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.weights = workflow_layer_weights_int16[conv5_2_factor];
            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.biases = workflow_layer_biases_int32[conv5_2_factor];

            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.center_offset[0] = 1;
            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.center_offset[1] = 1;

            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.stride[0] = 1;
            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.stride[1] = 1;

            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.accumulator = nnwrkld_accumulator_fraction[conv5_2_factor];
            workflow_layer[conv5_2]->arguments.forward_convolution_int16_fixedpoint.activation.fractions.output = nnwrkld_output_fraction[conv5_2_factor];

            workflow_layer[conv5_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv5_2]->output_format[0].format_3d = { { 13, 13, 256 / 2 } };
        }

        // merge g1 and g2
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = {{workflow_layer[conv5_1],0},{workflow_layer[conv5_2],0}};
            di->workflow_item_create_function(&workflow_layer[merge5], 2, inputs_descriptor, 1);

            workflow_layer[merge5]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_layer[merge5]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_layer[merge5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[merge5]->output_format[0].format_3d = { { 13, 13, 256 } };
        }

        // maxpool: 3x3 stride 2x2;
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[merge5], 0 };
            di->workflow_item_create_function(&workflow_layer[pool5], 1, &inputs_descriptor, 1); // pooling

            workflow_layer[pool5]->type = NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT;
            workflow_layer[pool5]->name = "p5";

            workflow_layer[pool5]->arguments.forward_pooling_fixedpoint.pool_size[0] = 3;
            workflow_layer[pool5]->arguments.forward_pooling_fixedpoint.pool_size[1] = 3;

            workflow_layer[pool5]->arguments.forward_pooling_fixedpoint.pool_stride[0] = 2;
            workflow_layer[pool5]->arguments.forward_pooling_fixedpoint.pool_stride[1] = 2;

            workflow_layer[pool5]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.accumulator = 16;
            workflow_layer[pool5]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.output = 8;

            workflow_layer[pool5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[pool5]->output_format[0].format_3d = { { 6, 6, 256 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 06
        //            full: ReLU
        //          output: 4096
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[pool5], 0 };
            di->workflow_item_create_function(&workflow_layer[fc6], 1, &inputs_descriptor, 1);

            workflow_layer[fc6]->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN;
            workflow_layer[fc6]->name = "fc6";

            workflow_layer[fc6]->arguments.fully_connected_forward_i16qn_i16qn.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;

            workflow_layer[fc6]->arguments.fully_connected_forward_i16qn_i16qn.weights = workflow_layer_weights_int16[fc6_factor];
            workflow_layer[fc6]->arguments.fully_connected_forward_i16qn_i16qn.biases = workflow_layer_biases_int32[fc6_factor];

            workflow_layer[fc6]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.accumulator = nnwrkld_accumulator_fraction[fc6_factor];
            workflow_layer[fc6]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.output = nnwrkld_output_fraction[fc6_factor];

            workflow_layer[fc6]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[fc6]->output_format[0].format_1d = { { 4096 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 07
        //            full: ReLU
        //          output: 4096
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[fc6], 0 };
            di->workflow_item_create_function(&workflow_layer[fc7], 1, &inputs_descriptor, 1);

            workflow_layer[fc7]->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN;
            workflow_layer[fc7]->name = "fc7";

            workflow_layer[fc7]->arguments.fully_connected_forward_i16qn_i16qn.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;

            workflow_layer[fc7]->arguments.fully_connected_forward_i16qn_i16qn.weights = workflow_layer_weights_int16[fc7_factor];
            workflow_layer[fc7]->arguments.fully_connected_forward_i16qn_i16qn.biases = workflow_layer_biases_int32[fc7_factor];

            workflow_layer[fc7]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.accumulator = nnwrkld_accumulator_fraction[fc7_factor];
            workflow_layer[fc7]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.output = nnwrkld_output_fraction[fc7_factor];

            workflow_layer[fc7]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[fc7]->output_format[0].format_1d = { { 4096 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 08
        //            full: ;
        //          output: 1000
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[fc7], 0 };
            di->workflow_item_create_function(&workflow_layer[fc8], 1, &inputs_descriptor, 1);

            workflow_layer[fc8]->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN;
            workflow_layer[fc8]->name = "fc8";

            workflow_layer[fc8]->arguments.fully_connected_forward_i16qn_i32qn.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_NONE;

            workflow_layer[fc8]->arguments.fully_connected_forward_i16qn_i32qn.weights = workflow_layer_weights_int16[fc8_factor];
            workflow_layer[fc8]->arguments.fully_connected_forward_i16qn_i32qn.biases = workflow_layer_biases_int32[fc8_factor];

            workflow_layer[fc8]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.accumulator = nnwrkld_accumulator_fraction[fc8_factor];
            workflow_layer[fc8]->arguments.fully_connected_forward_i16qn_i32qn.activation.fractions.output = nnwrkld_output_fraction[fc8_factor];

            workflow_layer[fc8]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[fc8]->output_format[0].format_1d = { { 1000 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 09 (softmax)
        //          output: 1000
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[fc8], 0 };
            di->workflow_item_create_function(&workflow_layer[softmax], 1, &inputs_descriptor, 1);

            workflow_layer[softmax]->type = NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT;

            workflow_layer[softmax]->arguments.forward_softmax_fixedpoint.input_fraction = nnwrkld_output_fraction[fc8_factor];

            workflow_layer[softmax]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[softmax]->output_format[0].format_1d = { { 1000 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 10 (output)
        //          output: 1000
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[softmax], 0 };
            di->workflow_item_create_function(&workflow_layer[output], 1, &inputs_descriptor, 1);

            workflow_layer[output]->type = NN_WORK_ITEM_TYPE_OUTPUT;

            workflow_layer[output]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[output]->output_format[0].format_1d = { { 1000 } };

        }

        // -------------------------------------------------------------------------------------------
        // END of workflow stages definition
        // -------------------------------------------------------------------------------------------
        workflow->input[0] = workflow_layer[input];
        workflow->output[0] = workflow_layer[output];
        // -------------------------------------------------------------------------------------------

        return workflow;

    }

    void cleanup() {

        if(!is_valid()) throw std::runtime_error(error_);

        for(auto wl : workflow_layer)
                di->workflow_item_delete_function(wl);

        di->workflow_delete_function(workflow);

        for(auto wlwf : workflow_layer_weights_float)
            if(wlwf!=nullptr) delete wlwf;

        for(auto wlwi : workflow_layer_weights_int16)
            if(wlwi!=nullptr) delete wlwi;

        for(auto wlbi : workflow_layer_biases_int32)
            if(wlbi!=nullptr) delete wlbi;

        for(auto wlbf : workflow_layer_biases_float)
            if(wlbf!=nullptr) delete wlbf;

        if(mean_factor!=nullptr) delete mean_factor;
    }
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran befor main execution starts.
// The sole function of this construction is attaching this workflow builder to
// library of workflow builders (singleton command pattern).
namespace {
    struct attach {
        workflow_for_test_caffenet_int16_learned test_workflow;
        attach() {
            workflows_for_tests::instance().add("caffenet_int16_learned", &test_workflow);
        }
    };

    attach attach_;
}
