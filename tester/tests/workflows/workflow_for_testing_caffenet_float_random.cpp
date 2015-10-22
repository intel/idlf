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
    mean_factor,
    conv1_weights,
    conv1_biases,
    conv2_1_weights,
    conv2_1_biases,
    conv2_2_weights,
    conv2_2_biases,
    conv3_weights,
    conv3_biases,
    conv4_1_weights,
    conv4_1_biases,
    conv4_2_weights,
    conv4_2_biases,
    conv5_1_weights,
    conv5_1_biases,
    conv5_2_weights,
    conv5_2_biases,
    fc6_weights,
    fc6_biases,
    fc7_weights,
    fc7_biases,
    fc8_weights,
    fc8_biases,
    last_weights_and_biases = fc8_biases
};


class workflow_for_test_caffenet_float_random: public workflows_for_tests_base
{

public:
    workflow_for_test_caffenet_float_random() { }
    bool is_valid() { return error_.empty(); }

private:
    std::string error_;

    uint16_t img_size = 227;

    // pointers to successive workflow parts
    nn_workflow_item_t        *workflow_layer[last_workflow_item+1];

    // pointers to nn_datas containing weights and biases;
    nn::data<float>           *workflow_layer_factor[last_weights_and_biases+1];

    nn_workflow_t             *workflow=nullptr;
    nn_device_interface_0_t   *di;

public:

    virtual nn_workflow_t *init_test_workflow(nn_device_interface_0_t *_di) {

        if(!is_valid()) throw std::runtime_error(error_);

        for(auto wi : workflow_layer) wi = nullptr;
        for(auto wb : workflow_layer_factor) wb = nullptr;

        this->di = _di;

        // create and populate nn:data factors (weights and biases) for successive layers

        workflow_layer_factor[mean_factor] = new nn::data<float>(img_size,img_size,3);
        nn_data_populate(workflow_layer_factor[mean_factor],104.007f,122.679f);

        workflow_layer_factor[conv1_weights] = new nn::data<float>(11,11,3,96);
        nn_data_populate(workflow_layer_factor[conv1_weights],-0.374f,0.403f);

        workflow_layer_factor[conv1_biases] = new nn::data<float>(96);
        nn_data_populate(workflow_layer_factor[conv1_biases],-0.854f,0.232f);

        workflow_layer_factor[conv2_1_weights] = new nn::data<float>(5,5,48,128);
        nn_data_populate(workflow_layer_factor[conv2_1_weights],-0.285f,0.379f);

        workflow_layer_factor[conv2_1_biases] = new nn::data<float>(128);
        nn_data_populate(workflow_layer_factor[conv2_1_biases],0.974f,1.034f);

        workflow_layer_factor[conv2_2_weights] = new nn::data<float>(5,5,48,128);
        nn_data_populate(workflow_layer_factor[conv2_2_weights],-0.269f,0.416f);

        workflow_layer_factor[conv2_2_biases] = new nn::data<float>(128);
        nn_data_populate(workflow_layer_factor[conv2_2_biases],0.958f,1.027f);

        workflow_layer_factor[conv3_weights] = new nn::data<float>(3,3,256,384);
        nn_data_populate(workflow_layer_factor[conv3_weights],-0.185f,0.512f);

        workflow_layer_factor[conv3_biases] = new nn::data<float>(384);
        nn_data_populate(workflow_layer_factor[conv3_biases],-0.104f,0.093f);

        workflow_layer_factor[conv4_1_weights] = new nn::data<float>(3,3,192,192);
        nn_data_populate(workflow_layer_factor[conv4_1_weights],-0.103f,0.322f);

        workflow_layer_factor[conv4_1_biases] = new nn::data<float>(192);
        nn_data_populate(workflow_layer_factor[conv4_1_biases],0.844f,1.142f);

        workflow_layer_factor[conv4_2_weights] = new nn::data<float>(3,3,192,192);
        nn_data_populate(workflow_layer_factor[conv4_2_weights],-0.142f,0.353f);

        workflow_layer_factor[conv4_2_biases] = new nn::data<float>(192);
        nn_data_populate(workflow_layer_factor[conv4_2_biases],0.77f,1.219f);

        workflow_layer_factor[conv5_1_weights] = new nn::data<float>(3,3,192,128);
        nn_data_populate(workflow_layer_factor[conv5_1_weights],-0.092f,0.254f);

        workflow_layer_factor[conv5_1_biases] = new nn::data<float>(128);
        nn_data_populate(workflow_layer_factor[conv5_1_biases],0.723f,1.50f);

        workflow_layer_factor[conv5_2_weights] = new nn::data<float>(3,3,192,128);
        nn_data_populate(workflow_layer_factor[conv5_2_weights],-0.133f,0.315f);

        workflow_layer_factor[conv5_2_biases] = new nn::data<float>(128);
        nn_data_populate(workflow_layer_factor[conv5_2_biases],0.623f,1.742f);

        workflow_layer_factor[fc6_weights] = new nn::data<float>(6,6,256,4096);
        nn_data_populate(workflow_layer_factor[fc6_weights],-0.035f,0.048f);

        workflow_layer_factor[fc6_biases] = new nn::data<float>(4096);
        nn_data_populate(workflow_layer_factor[fc6_biases],0.92f,1.057f);

        workflow_layer_factor[fc7_weights] = new nn::data<float>(4096,4096);
        nn_data_populate(workflow_layer_factor[fc7_weights],-0.032f,0.052f);

        workflow_layer_factor[fc7_biases] = new nn::data<float>(4096);
        nn_data_populate(workflow_layer_factor[fc7_biases],0.741f,1.26f);

        workflow_layer_factor[fc8_weights] = new nn::data<float>(4096,1000);
        nn_data_populate(workflow_layer_factor[fc8_weights],-0.045f,0.067f);

        workflow_layer_factor[fc8_biases] = new nn::data<float>(1000);
        nn_data_populate(workflow_layer_factor[fc8_biases],-0.351f,0.425f);

        di->workflow_create_function(&workflow,1,1);
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
            workflow_layer[mean_substract]->arguments.forward_arithmetic.factor = workflow_layer_factor[mean_factor];
            workflow_layer[mean_substract]->arguments.forward_arithmetic.arithmetic_function = NN_ARITHMETIC_FUNCTION_SUBTRACTION;

            workflow_layer[mean_substract]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[mean_substract]->output_format[0].format_3d ={{img_size,img_size,3}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 01
        //           convo: 11x11 stride 4x4; ReLU; output: 55x55x96
        //         maxpool: 3x3 stride 2x2;
        //            norm: RESPONSE_ACROSS_MAPS
        //          output: 27x27x96
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[mean_substract],0};
            di->workflow_item_create_function(&workflow_layer[conv1],1,&inputs_descriptor,1);

            workflow_layer[conv1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv1]->name = "c1";

            workflow_layer[conv1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_layer[conv1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;

            workflow_layer[conv1]->arguments.forward_convolution.weights = workflow_layer_factor[conv1_weights];
            workflow_layer[conv1]->arguments.forward_convolution.biases = workflow_layer_factor[conv1_biases];

            workflow_layer[conv1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_layer[conv1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_layer[conv1]->arguments.forward_convolution.stride[0] = 4;
            workflow_layer[conv1]->arguments.forward_convolution.stride[1] = 4;

            workflow_layer[conv1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv1]->output_format[0].format_3d ={{55,55,96}};
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[conv1],0};
            di->workflow_item_create_function(&workflow_layer[pool1],1,&inputs_descriptor,1);

            workflow_layer[pool1]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_layer[pool1]->name = "p1";

            workflow_layer[pool1]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_layer[pool1]->arguments.forward_pooling.size[0] = 3;
            workflow_layer[pool1]->arguments.forward_pooling.size[1] = 3;
            workflow_layer[pool1]->arguments.forward_pooling.stride[0] = 2;
            workflow_layer[pool1]->arguments.forward_pooling.stride[1] = 2;

            workflow_layer[pool1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[pool1]->output_format[0].format_3d ={{27,27,96}};
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[pool1],0};
            di->workflow_item_create_function(&workflow_layer[norm1],1,&inputs_descriptor,1);

            workflow_layer[norm1]->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            workflow_layer[norm1]->name = "lrn1";

            workflow_layer[norm1]->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            workflow_layer[norm1]->arguments.forward_normalization.normalization.k = 1; // in Krishevsky's article is 2
            workflow_layer[norm1]->arguments.forward_normalization.normalization.n = 5;
            workflow_layer[norm1]->arguments.forward_normalization.normalization.alpha = 0.0001f/5; // in Krishevsky's paper is 1e-4,
                                                                                                    // but didn't write that sum of the squares
                                                                                                    // is divided by number of elements (n)
            workflow_layer[norm1]->arguments.forward_normalization.normalization.beta = 0.75f;

            workflow_layer[norm1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[norm1]->output_format[0].format_3d ={{27,27,96}};
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
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[norm1],0};
            di->workflow_item_create_function(&workflow_layer[subv1_1],1,&inputs_descriptor,1); // view g1

            workflow_layer[subv1_1]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv1_1]->arguments.view.origin[0] = 0;
            workflow_layer[subv1_1]->arguments.view.origin[1] = 0;
            workflow_layer[subv1_1]->arguments.view.origin[2] = 0;

            workflow_layer[subv1_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv1_1]->output_format[0].format_3d ={{27,27,96/2}};

        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[norm1],0};
            di->workflow_item_create_function(&workflow_layer[subv1_2],1,&inputs_descriptor,1);   // view g2

            workflow_layer[subv1_2]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv1_2]->arguments.view.origin[0] = 0;
            workflow_layer[subv1_2]->arguments.view.origin[1] = 0;
            workflow_layer[subv1_2]->arguments.view.origin[2] = (96/2);

            workflow_layer[subv1_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv1_2]->output_format[0].format_3d ={{27,27,96/2}};
        }

        // convolution 2, g1: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[subv1_1],0};
            di->workflow_item_create_function(&workflow_layer[conv2_1],1,&inputs_descriptor,1);

            workflow_layer[conv2_1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv2_1]->name = "c2g1";

            workflow_layer[conv2_1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv2_1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv2_1]->arguments.forward_convolution.weights = workflow_layer_factor[conv2_1_weights];
            workflow_layer[conv2_1]->arguments.forward_convolution.biases = workflow_layer_factor[conv2_1_biases];

            workflow_layer[conv2_1]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_layer[conv2_1]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_layer[conv2_1]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv2_1]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv2_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv2_1]->output_format[0].format_3d ={{27,27,256/2}};
        }

        // convolution 2, g2: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[subv1_2],0};
            di->workflow_item_create_function(&workflow_layer[conv2_2],1,&inputs_descriptor,1);

            workflow_layer[conv2_2]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv2_2]->name = "c2g2";

            workflow_layer[conv2_2]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv2_2]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv2_2]->arguments.forward_convolution.weights = workflow_layer_factor[conv2_2_weights];
            workflow_layer[conv2_2]->arguments.forward_convolution.biases = workflow_layer_factor[conv2_2_biases];

            workflow_layer[conv2_2]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_layer[conv2_2]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_layer[conv2_2]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv2_2]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv2_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv2_2]->output_format[0].format_3d ={{27,27,256/2}};
        }

        // merge g1 and g2
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] ={{workflow_layer[conv2_1],0},{workflow_layer[conv2_2],0}};
            di->workflow_item_create_function(&workflow_layer[merge2],2,inputs_descriptor,1);

            workflow_layer[merge2]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_layer[merge2]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_layer[merge2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[merge2]->output_format[0].format_3d ={{27,27,256}};

        }

        // maxpool: 3x3 stride 2x2;
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[merge2],0};
            di->workflow_item_create_function(&workflow_layer[pool2],1,&inputs_descriptor,1); // pooling

            workflow_layer[pool2]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_layer[pool2]->name = "p2";

            workflow_layer[pool2]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

            workflow_layer[pool2]->arguments.forward_pooling.size[0] = 3;
            workflow_layer[pool2]->arguments.forward_pooling.size[1] = 3;

            workflow_layer[pool2]->arguments.forward_pooling.stride[0] = 2;
            workflow_layer[pool2]->arguments.forward_pooling.stride[1] = 2;

            workflow_layer[pool2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[pool2]->output_format[0].format_3d ={{13,13,256}};
        }

        //norm: RESPONSE_ACROSS_MAPS; output: 13x13x256
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[pool2],0};
            di->workflow_item_create_function(&workflow_layer[norm2],1,&inputs_descriptor,1);

            workflow_layer[norm2]->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            workflow_layer[norm2]->name = "lrn2";

            workflow_layer[norm2]->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            workflow_layer[norm2]->arguments.forward_normalization.normalization.k = 1;              // |
            workflow_layer[norm2]->arguments.forward_normalization.normalization.n = 5;              // |
            workflow_layer[norm2]->arguments.forward_normalization.normalization.alpha = 0.0001f/5;  // > see coment at wrkflwi_stage_1_norm
            workflow_layer[norm2]->arguments.forward_normalization.normalization.beta = 0.75f;       // |

            workflow_layer[norm2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[norm2]->output_format[0].format_3d ={{13,13,256}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 03
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x384
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[norm2],0};
            di->workflow_item_create_function(&workflow_layer[conv3],1,&inputs_descriptor,1);

            workflow_layer[conv3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv3]->name = "c3";
            workflow_layer[conv3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv3]->arguments.forward_convolution.weights = workflow_layer_factor[conv3_weights];
            workflow_layer[conv3]->arguments.forward_convolution.biases = workflow_layer_factor[conv3_biases];

            workflow_layer[conv3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_layer[conv3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_layer[conv3]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv3]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv3]->output_format[0].format_3d ={{13,13,384}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 04
        //           split: 2 (z-axis 384/2)
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x(2*384/2) (continue split to next stage)
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[conv3],0};
            di->workflow_item_create_function(&workflow_layer[subv3_1],1,&inputs_descriptor,1); // view g1

            workflow_layer[subv3_1]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv3_1]->arguments.view.origin[0] = 0;
            workflow_layer[subv3_1]->arguments.view.origin[1] = 0;
            workflow_layer[subv3_1]->arguments.view.origin[2] = 0;

            workflow_layer[subv3_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv3_1]->output_format[0].format_3d ={{13,13,384/2}};
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[conv3],0};
            di->workflow_item_create_function(&workflow_layer[subv3_2],1,&inputs_descriptor,1); // view g2

            workflow_layer[subv3_2]->type = NN_WORK_ITEM_TYPE_VIEW;
            workflow_layer[subv3_2]->arguments.view.origin[0] = 0;
            workflow_layer[subv3_2]->arguments.view.origin[1] = 0;
            workflow_layer[subv3_2]->arguments.view.origin[2] = 384/2;

            workflow_layer[subv3_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[subv3_2]->output_format[0].format_3d ={{13,13,384/2}};

        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[subv3_1],0};
            di->workflow_item_create_function(&workflow_layer[conv4_1],1,&inputs_descriptor,1); // conv g1

            workflow_layer[conv4_1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv4_1]->name = "c4g1";

            workflow_layer[conv4_1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv4_1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv4_1]->arguments.forward_convolution.weights = workflow_layer_factor[conv4_1_weights];
            workflow_layer[conv4_1]->arguments.forward_convolution.biases = workflow_layer_factor[conv4_1_biases];

            workflow_layer[conv4_1]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_layer[conv4_1]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_layer[conv4_1]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv4_1]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv4_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv4_1]->output_format[0].format_3d ={{13,13,384/2}};
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[subv3_2],0};
            di->workflow_item_create_function(&workflow_layer[conv4_2],1,&inputs_descriptor,1); // conv g2

            workflow_layer[conv4_2]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv4_2]->name = "c4g2";

            workflow_layer[conv4_2]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv4_2]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv4_2]->arguments.forward_convolution.weights = workflow_layer_factor[conv4_1_weights];
            workflow_layer[conv4_2]->arguments.forward_convolution.biases = workflow_layer_factor[conv4_2_biases];

            workflow_layer[conv4_2]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_layer[conv4_2]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_layer[conv4_2]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv4_2]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv4_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv4_2]->output_format[0].format_3d ={{13,13,384/2}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 05
        //           convo: 3x3 stride 1x1; ReLU; 0-padded; output: 13x13x(2*256/2)
        //           merge: (z-axis)
        //         maxpool: 3x3 stride 2x2;
        //          output: 13x13x256
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[conv4_1],0};
            di->workflow_item_create_function(&workflow_layer[conv5_1],1,&inputs_descriptor,1); // conv g1

            workflow_layer[conv5_1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv5_1]->name = "c5g1";

            workflow_layer[conv5_1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv5_1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv5_1]->arguments.forward_convolution.weights = workflow_layer_factor[conv5_1_weights];
            workflow_layer[conv5_1]->arguments.forward_convolution.biases = workflow_layer_factor[conv5_1_biases];

            workflow_layer[conv5_1]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_layer[conv5_1]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_layer[conv5_1]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv5_1]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv5_1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv5_1]->output_format[0].format_3d ={{13,13,256/2}};
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[conv4_2],0};
            di->workflow_item_create_function(&workflow_layer[conv5_2],1,&inputs_descriptor,1); // conv g2

            workflow_layer[conv5_2]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv5_2]->name = "c5g2";

            workflow_layer[conv5_2]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv5_2]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv5_2]->arguments.forward_convolution.weights = workflow_layer_factor[conv5_2_weights];
            workflow_layer[conv5_2]->arguments.forward_convolution.biases = workflow_layer_factor[conv5_2_biases];

            workflow_layer[conv5_2]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_layer[conv5_2]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_layer[conv5_2]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv5_2]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv5_2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv5_2]->output_format[0].format_3d ={{13,13,256/2}};
        }

        // merge g1 and g2
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] ={{workflow_layer[conv5_1],0},{workflow_layer[conv5_2],0}};
            di->workflow_item_create_function(&workflow_layer[merge5],2,inputs_descriptor,1);

            workflow_layer[merge5]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_layer[merge5]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_layer[merge5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[merge5]->output_format[0].format_3d ={{13,13,256}};
        }

        // maxpool: 3x3 stride 2x2;
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[merge5],0};
            di->workflow_item_create_function(&workflow_layer[pool5],1,&inputs_descriptor,1); // pooling

            workflow_layer[pool5]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_layer[pool5]->name = "p5";

            workflow_layer[pool5]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

            workflow_layer[pool5]->arguments.forward_pooling.size[0] = 3;
            workflow_layer[pool5]->arguments.forward_pooling.size[1] = 3;

            workflow_layer[pool5]->arguments.forward_pooling.stride[0] = 2;
            workflow_layer[pool5]->arguments.forward_pooling.stride[1] = 2;

            workflow_layer[pool5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[pool5]->output_format[0].format_3d ={{6,6,256}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 06
        //            full: ReLU
        //          output: 4096
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[pool5],0};
            di->workflow_item_create_function(&workflow_layer[fc6],1,&inputs_descriptor,1);

            workflow_layer[fc6]->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            workflow_layer[fc6]->name = "fc6";

            workflow_layer[fc6]->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_RELU;

            workflow_layer[fc6]->arguments.forward_fully_connected.weights = workflow_layer_factor[fc6_weights];
            workflow_layer[fc6]->arguments.forward_fully_connected.biases = workflow_layer_factor[fc6_biases];

            workflow_layer[fc6]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[fc6]->output_format[0].format_1d ={{4096}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 07
        //            full: ReLU
        //          output: 4096
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[fc6],0};
            di->workflow_item_create_function(&workflow_layer[fc7],1,&inputs_descriptor,1);

            workflow_layer[fc7]->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            workflow_layer[fc7]->name = "fc7";
            workflow_layer[fc7]->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_RELU;

            workflow_layer[fc7]->arguments.forward_fully_connected.weights = workflow_layer_factor[fc7_weights];
            workflow_layer[fc7]->arguments.forward_fully_connected.biases = workflow_layer_factor[fc7_biases];

            workflow_layer[fc7]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[fc7]->output_format[0].format_1d ={{4096}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 08
        //            full: ;
        //          output: 1000
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[fc7],0};
            di->workflow_item_create_function(&workflow_layer[fc8],1,&inputs_descriptor,1);

            workflow_layer[fc8]->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            workflow_layer[fc8]->name = "fc8";

            workflow_layer[fc8]->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            workflow_layer[fc8]->arguments.forward_fully_connected.weights = workflow_layer_factor[fc8_weights];
            workflow_layer[fc8]->arguments.forward_fully_connected.biases = workflow_layer_factor[fc8_biases];

            workflow_layer[fc8]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[fc8]->output_format[0].format_1d ={{1000}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 09 (softmax)
        //          output: 1000
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[fc8],0};
            di->workflow_item_create_function(&workflow_layer[softmax],1,&inputs_descriptor,1);

            workflow_layer[softmax]->type = NN_WORK_ITEM_TYPE_SOFTMAX;

            workflow_layer[softmax]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[softmax]->output_format[0].format_1d ={{1000}};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 10 (output)
        //          output: 1000
        {
            nn_workflow_use_descriptor_t inputs_descriptor ={workflow_layer[softmax],0};
            di->workflow_item_create_function(&workflow_layer[output],1,&inputs_descriptor,1);

            workflow_layer[output]->type = NN_WORK_ITEM_TYPE_OUTPUT;

            workflow_layer[output]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[output]->output_format[0].format_1d ={{1000}};

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

        for(auto wb : workflow_layer_factor)
            if(wb!=nullptr) delete wb;

    }
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran befor main execution starts.
// The sole function of this construction is attaching this workflow builder to
// library of workflow builders (singleton command pattern).
namespace {
    struct attach {
        workflow_for_test_caffenet_float_random test_workflow;
        attach() {
            workflows_for_tests::instance().add("caffenet_float_random", &test_workflow);
        }
    };

    attach attach_;
}
