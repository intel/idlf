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

#include "workflow_builder.h"
#include "nn_data_tools.h"

class workflow_builder_caffenet_float: public workflow_builder_base
{

public:

    workflow_builder_caffenet_float() : workflow_builder_base(227) {
        RGB_order = false;
        image_process = fi::resize_image_to_square;
        try {
            read_file_to_vector(labels, "weights_caffenet/names.txt", false);
            read_file_to_vector(wwids, "weights_caffenet/wwids.txt", false);
        }
        catch(std::runtime_error &e) {
            error_ = e.what();
        }
    }

    bool is_valid() { return error_.empty(); }
private:
    std::string error_;

    // pointers to successive workflow parts
    nn_workflow_item_t
        *wrkflwi_input,
        *wrkflwi_stage_0_mean_substract,
        *wrkflwi_stage_1_conv,
        *wrkflwi_stage_1_pool,
        *wrkflwi_stage_1_norm,
        *wrkflwi_stage_1_2_g1_subv,
        *wrkflwi_stage_1_2_g2_subv,
        *wrkflwi_stage_2_g1_conv,
        *wrkflwi_stage_2_g2_conv,
        *wrkflwi_stage_2_merge,
        *wrkflwi_stage_2_pool,
        *wrkflwi_stage_2_norm,
        *wrkflwi_stage_3_conv,
        *wrkflwi_stage_3_4_g1_subv,
        *wrkflwi_stage_3_4_g2_subv,
        *wrkflwi_stage_4_g1_conv,
        *wrkflwi_stage_4_g2_conv,
        *wrkflwi_stage_5_g1_conv,
        *wrkflwi_stage_5_g2_conv,
        *wrkflwi_stage_5_merge,
        *wrkflwi_stage_5_pool,
        *wrkflwi_stage_6_fc,
        *wrkflwi_stage_7_fc,
        *wrkflwi_stage_8_fc,
        *wrkflwi_softmax,
        *wrkflwi_output;

    // pointers to <nn_workload_data>s containing weights and biases;
    nn::data<float>
        *nnwrkld_imagenet_mean,
        *nnwrkld_conv1_weights,
        *nnwrkld_conv1_biases,
        *nnwrkld_conv2_g1_weights,
        *nnwrkld_conv2_g1_biases,
        *nnwrkld_conv2_g2_weights,
        *nnwrkld_conv2_g2_biases,
        *nnwrkld_conv3_weights,
        *nnwrkld_conv3_biases,
        *nnwrkld_conv4_g1_weights,
        *nnwrkld_conv4_g1_biases,
        *nnwrkld_conv4_g2_weights,
        *nnwrkld_conv4_g2_biases,
        *nnwrkld_conv5_g1_weights,
        *nnwrkld_conv5_g1_biases,
        *nnwrkld_conv5_g2_weights,
        *nnwrkld_conv5_g2_biases,
        *nnwrkld_fc6_weights,
        *nnwrkld_fc6_biases,
        *nnwrkld_fc7_weights,
        *nnwrkld_fc7_biases,
        *nnwrkld_fc8_weights,
        *nnwrkld_fc8_biases;

    nn_workflow_t           *workflow;
    nn_device_interface_0_t *di;

public:

    void cleanup(){
        if(!is_valid()) throw std::runtime_error(error_);

        /* ****************************************************************************************** */
        /* Cleanup in memory                                                                          */
        /* ****************************************************************************************** */
        std::cout
            << "Cleanup in memory"
            << std::endl
            << "========================================================"
            << std::endl;

        di->workflow_item_delete_function(wrkflwi_input);
        di->workflow_item_delete_function(wrkflwi_stage_0_mean_substract);
        di->workflow_item_delete_function(wrkflwi_stage_1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_1_pool);
        di->workflow_item_delete_function(wrkflwi_stage_1_norm);
        di->workflow_item_delete_function(wrkflwi_stage_1_2_g1_subv);
        di->workflow_item_delete_function(wrkflwi_stage_1_2_g2_subv);
        di->workflow_item_delete_function(wrkflwi_stage_2_g1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_2_g2_conv);
        di->workflow_item_delete_function(wrkflwi_stage_2_merge);
        di->workflow_item_delete_function(wrkflwi_stage_2_pool);
        di->workflow_item_delete_function(wrkflwi_stage_2_norm);
        di->workflow_item_delete_function(wrkflwi_stage_3_conv);
        di->workflow_item_delete_function(wrkflwi_stage_3_4_g1_subv);
        di->workflow_item_delete_function(wrkflwi_stage_3_4_g2_subv);
        di->workflow_item_delete_function(wrkflwi_stage_4_g1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_4_g2_conv);
        di->workflow_item_delete_function(wrkflwi_stage_5_g1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_5_g2_conv);
        di->workflow_item_delete_function(wrkflwi_stage_5_merge);
        di->workflow_item_delete_function(wrkflwi_stage_5_pool);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc);
        di->workflow_item_delete_function(wrkflwi_stage_8_fc);
        di->workflow_item_delete_function(wrkflwi_softmax);
        di->workflow_item_delete_function(wrkflwi_output);

        di->workflow_delete_function(workflow);

        delete nnwrkld_imagenet_mean;
        delete nnwrkld_conv1_weights;
        delete nnwrkld_conv1_biases;
        delete nnwrkld_conv2_g1_weights;
        delete nnwrkld_conv2_g1_biases;
        delete nnwrkld_conv2_g2_weights;
        delete nnwrkld_conv2_g2_biases;
        delete nnwrkld_conv3_weights;
        delete nnwrkld_conv3_biases;
        delete nnwrkld_conv4_g1_weights;
        delete nnwrkld_conv4_g1_biases;
        delete nnwrkld_conv4_g2_weights;
        delete nnwrkld_conv4_g2_biases;
        delete nnwrkld_conv5_g1_weights;
        delete nnwrkld_conv5_g1_biases;
        delete nnwrkld_conv5_g2_weights;
        delete nnwrkld_conv5_g2_biases;
        delete nnwrkld_fc6_weights;
        delete nnwrkld_fc6_biases;
        delete nnwrkld_fc7_weights;
        delete nnwrkld_fc7_biases;
        delete nnwrkld_fc8_weights;
        delete nnwrkld_fc8_biases;
        delete di;
    }

    virtual nn_workflow_t *init_workflow(nn_device_interface_0_t *di, uint32_t batch){

        if(!is_valid()) throw std::runtime_error(error_);

        this->di = di;

        std::cout
            << "--------------------------------------------------------"
            << std::endl
            << "Loading weights and biases"
            << std::endl << std::endl;

        // Load weights and biases
        auto load_biases_or_weights = [](std::string wb_file_name) {
            nn::data<float> *wb_pointer = nn_data_load_from_file_time_measure(wb_file_name);
            if(wb_pointer == nullptr) {
                std::cerr << "Can't load " << wb_file_name << std::endl;
                throw;
            }
            return wb_pointer;
        };

        try {
            nnwrkld_imagenet_mean  = load_biases_or_weights("weights_caffenet/imagenet_mean.nnd");
            nnwrkld_conv1_weights  = load_biases_or_weights("weights_caffenet/conv1.nnd");
            nnwrkld_conv1_biases  = load_biases_or_weights("weights_caffenet/conv1_bias.nnd");
            nnwrkld_conv2_g1_weights  = load_biases_or_weights("weights_caffenet/conv2_g1.nnd");
            nnwrkld_conv2_g1_biases  = load_biases_or_weights("weights_caffenet/conv2_bias_g1.nnd");
            nnwrkld_conv2_g2_weights  = load_biases_or_weights("weights_caffenet/conv2_g2.nnd");
            nnwrkld_conv2_g2_biases  = load_biases_or_weights("weights_caffenet/conv2_bias_g2.nnd");
            nnwrkld_conv3_weights  = load_biases_or_weights("weights_caffenet/conv3.nnd");
            nnwrkld_conv3_biases  = load_biases_or_weights("weights_caffenet/conv3_bias.nnd");
            nnwrkld_conv4_g1_weights  = load_biases_or_weights("weights_caffenet/conv4_g1.nnd");
            nnwrkld_conv4_g1_biases  = load_biases_or_weights("weights_caffenet/conv4_bias_g1.nnd");
            nnwrkld_conv4_g2_weights  = load_biases_or_weights("weights_caffenet/conv4_g2.nnd");
            nnwrkld_conv4_g2_biases  = load_biases_or_weights("weights_caffenet/conv4_bias_g2.nnd");
            nnwrkld_conv5_g1_weights  = load_biases_or_weights("weights_caffenet/conv5_g1.nnd");
            nnwrkld_conv5_g1_biases  = load_biases_or_weights("weights_caffenet/conv5_bias_g1.nnd");
            nnwrkld_conv5_g2_weights  = load_biases_or_weights("weights_caffenet/conv5_g2.nnd");
            nnwrkld_conv5_g2_biases  = load_biases_or_weights("weights_caffenet/conv5_bias_g2.nnd");
            nnwrkld_fc6_weights  = load_biases_or_weights("weights_caffenet/fc6.nnd");
            nnwrkld_fc6_biases  = load_biases_or_weights("weights_caffenet/fc6_bias.nnd");
            nnwrkld_fc7_weights  = load_biases_or_weights("weights_caffenet/fc7.nnd");
            nnwrkld_fc7_biases  = load_biases_or_weights("weights_caffenet/fc7_bias.nnd");
            nnwrkld_fc8_weights  = load_biases_or_weights("weights_caffenet/fc8.nnd");
            nnwrkld_fc8_biases  = load_biases_or_weights("weights_caffenet/fc8_bias.nnd");
        }
        catch(...) {
            return workflow;
        }

        std::cout
            << "--------------------------------------------------------" << std::endl
            << "Build of workflow" << std::endl;

        di->workflow_create_function(&workflow, 1, 1);

        // ------------------------------------------------------------------------------------------
        // STAGE 0 (input)
        //         output: 227x227x3

        di->workflow_item_create_function(&wrkflwi_input, 0, nullptr);
        {
            wrkflwi_input->type = NN_WORK_ITEM_TYPE_INPUT;
            wrkflwi_input->arguments.input.index = 0;
            wrkflwi_input->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_input->output_format.format_3d ={ { img_size, img_size, 3 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 0 (imagenet_mean_subtract)
        //         output: 227x227x3
        di->workflow_item_create_function(&wrkflwi_stage_0_mean_substract, 1, &wrkflwi_input);
        {
            wrkflwi_stage_0_mean_substract->type = NN_WORK_ITEM_TYPE_ARITHMETIC;
            wrkflwi_stage_0_mean_substract->arguments.forward_arithmetic.factor = nnwrkld_imagenet_mean;
            wrkflwi_stage_0_mean_substract->arguments.forward_arithmetic.arithmetic_function = NN_ARITHMETIC_FUNCTION_SUBTRACTION;

            wrkflwi_stage_0_mean_substract->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_0_mean_substract->output_format.format_3d ={ { img_size, img_size, 3 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 01
        //           convo: 11x11 stride 4x4; ReLU; output: 55x55x96
        //         maxpool: 3x3 stride 2x2;
        //            norm: RESPONSE_ACROSS_MAPS
        //          output: 27x27x96

        di->workflow_item_create_function(&wrkflwi_stage_1_conv, 1, &wrkflwi_stage_0_mean_substract);
        {
            wrkflwi_stage_1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_1_conv->name = "c1";

            wrkflwi_stage_1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            wrkflwi_stage_1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;

            wrkflwi_stage_1_conv->arguments.forward_convolution.weights = nnwrkld_conv1_weights;
            wrkflwi_stage_1_conv->arguments.forward_convolution.biases = nnwrkld_conv1_biases;

            wrkflwi_stage_1_conv->arguments.forward_convolution.center_offset[0] = 0;
            wrkflwi_stage_1_conv->arguments.forward_convolution.center_offset[1] = 0;

            wrkflwi_stage_1_conv->arguments.forward_convolution.stride[0] = 4;
            wrkflwi_stage_1_conv->arguments.forward_convolution.stride[1] = 4;

            wrkflwi_stage_1_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_1_conv->output_format.format_3d ={ { 55, 55, 96 } };
        }

        di->workflow_item_create_function(&wrkflwi_stage_1_pool, 1, &wrkflwi_stage_1_conv);
        {
            wrkflwi_stage_1_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_1_pool->name = "p1";

            wrkflwi_stage_1_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            wrkflwi_stage_1_pool->arguments.forward_pooling.size[0] = 3;
            wrkflwi_stage_1_pool->arguments.forward_pooling.size[1] = 3;
            wrkflwi_stage_1_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_1_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_1_pool->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_1_pool->output_format.format_3d ={ { 27, 27, 96 } };
        }

        di->workflow_item_create_function(&wrkflwi_stage_1_norm, 1, &wrkflwi_stage_1_pool);
        {
            wrkflwi_stage_1_norm->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            wrkflwi_stage_1_norm->name = "lrn1";

            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.k = 1; // in Krishevsky's article is 2
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.n = 5;
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.alpha = 0.0001f/5; // in Krishevsky's paper is 1e-4,
                                                                                                   // but didn't write that sum of the squares
                                                                                                   // is divided by number of elements (n)
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.beta = 0.75f;

            wrkflwi_stage_1_norm->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_1_norm->output_format.format_3d ={ { 27, 27, 96 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 02
        //           split: 2 (z-axis 96/2); output 27x27x(2*96/2)
        //           convo: 5x5 stride 1x1; ReLU; 0-padded output: 27x27x(2*256/2)
        //           merge: (z-axis)
        //         maxpool: 3x3 stride 2x2;
        //            norm: RESPONSE_ACROSS_MAPS
        //          output: 13x13x256

        di->workflow_item_create_function(&wrkflwi_stage_1_2_g1_subv, 1, &wrkflwi_stage_1_norm); // view g1
        {
            wrkflwi_stage_1_2_g1_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_1_2_g1_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_1_2_g1_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_1_2_g1_subv->arguments.view.origin[2] = 0;

            wrkflwi_stage_1_2_g1_subv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_1_2_g1_subv->output_format.format_3d ={ { 27, 27, 96/2 } };

        }
        di->workflow_item_create_function(&wrkflwi_stage_1_2_g2_subv, 1, &wrkflwi_stage_1_norm);   // view g2
        {
            wrkflwi_stage_1_2_g2_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_1_2_g2_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_1_2_g2_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_1_2_g2_subv->arguments.view.origin[2] = (96/2);

            wrkflwi_stage_1_2_g2_subv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_1_2_g2_subv->output_format.format_3d ={ { 27, 27, 96/2 } };
        }
        // convolution 2, g1: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        di->workflow_item_create_function(&wrkflwi_stage_2_g1_conv, 1, &wrkflwi_stage_1_2_g1_subv);
        {
            wrkflwi_stage_2_g1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_2_g1_conv->name = "c2g1";

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.weights = nnwrkld_conv2_g1_weights;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.biases = nnwrkld_conv2_g1_biases;

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.center_offset[0] = 2;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.center_offset[1] = 2;

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_2_g1_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_2_g1_conv->output_format.format_3d ={ { 27, 27, 256/2 } };
        }

        // convolution 2, g2: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        di->workflow_item_create_function(&wrkflwi_stage_2_g2_conv, 1, &wrkflwi_stage_1_2_g2_subv);
        {
            wrkflwi_stage_2_g2_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_2_g2_conv->name = "c2g2";

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.weights = nnwrkld_conv2_g2_weights;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.biases = nnwrkld_conv2_g2_biases;

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.center_offset[0] = 2;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.center_offset[1] = 2;

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_2_g2_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_2_g2_conv->output_format.format_3d ={ { 27, 27, 256/2 } };
        }

        // merge g1 and g2
        nn_workflow_item_t *input_merge_1[2] = { wrkflwi_stage_2_g1_conv, wrkflwi_stage_2_g2_conv };
        di->workflow_item_create_function(&wrkflwi_stage_2_merge, 2, input_merge_1);
        {
            wrkflwi_stage_2_merge->type = NN_WORK_ITEM_TYPE_MERGE;
            wrkflwi_stage_2_merge->arguments.forward_merge.axis = 2; // value 2 for z-axis

            wrkflwi_stage_2_merge->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_2_merge->output_format.format_3d ={ { 27, 27, 256 } };

        }

        // maxpool: 3x3 stride 2x2;
        di->workflow_item_create_function(&wrkflwi_stage_2_pool, 1, &wrkflwi_stage_2_merge); // pooling
        {
            wrkflwi_stage_2_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_2_pool->name = "p2";

            wrkflwi_stage_2_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

            wrkflwi_stage_2_pool->arguments.forward_pooling.size[0] = 3;
            wrkflwi_stage_2_pool->arguments.forward_pooling.size[1] = 3;

            wrkflwi_stage_2_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_2_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_2_pool->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_2_pool->output_format.format_3d ={ { 13, 13, 256 } };
        }

        //norm: RESPONSE_ACROSS_MAPS; output: 13x13x256
        di->workflow_item_create_function(&wrkflwi_stage_2_norm, 1, &wrkflwi_stage_2_pool);
        {
            wrkflwi_stage_2_norm->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            wrkflwi_stage_2_norm->name = "lrn2";

            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.k = 1;              // |
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.n = 5;              // |
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.alpha = 0.0001f/5;  // > see coment at wrkflwi_stage_1_norm
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.beta = 0.75f;       // |

            wrkflwi_stage_2_norm->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_2_norm->output_format.format_3d ={ { 13, 13, 256 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 03
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x384

        di->workflow_item_create_function(&wrkflwi_stage_3_conv, 1, &wrkflwi_stage_2_norm);
        {

            wrkflwi_stage_3_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_3_conv->name = "c3";
            wrkflwi_stage_3_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            wrkflwi_stage_3_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_3_conv->arguments.forward_convolution.weights = nnwrkld_conv3_weights;
            wrkflwi_stage_3_conv->arguments.forward_convolution.biases = nnwrkld_conv3_biases;

            wrkflwi_stage_3_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_3_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_3_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_3_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_3_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_3_conv->output_format.format_3d ={ { 13, 13, 384 } };
        }
        // ------------------------------------------------------------------------------------------
        // STAGE 04
        //           split: 2 (z-axis 384/2)
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x(2*384/2) (continue split to next stage)

        di->workflow_item_create_function(&wrkflwi_stage_3_4_g1_subv, 1, &wrkflwi_stage_3_conv); // view g1
        {
            wrkflwi_stage_3_4_g1_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_3_4_g1_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_3_4_g1_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_3_4_g1_subv->arguments.view.origin[2] = 0;

            wrkflwi_stage_3_4_g1_subv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_3_4_g1_subv->output_format.format_3d ={ { 13, 13, 384/2 } };
        }

        di->workflow_item_create_function(&wrkflwi_stage_3_4_g2_subv, 1, &wrkflwi_stage_3_conv); // view g2
        {
            wrkflwi_stage_3_4_g2_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_3_4_g2_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_3_4_g2_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_3_4_g2_subv->arguments.view.origin[2] = 384/2;

            wrkflwi_stage_3_4_g2_subv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_3_4_g2_subv->output_format.format_3d ={ { 13, 13, 384/2 } };

        }

        di->workflow_item_create_function(&wrkflwi_stage_4_g1_conv, 1, &wrkflwi_stage_3_4_g1_subv); // conv g1
        {
            wrkflwi_stage_4_g1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_4_g1_conv->name = "c4g1";

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.weights = nnwrkld_conv4_g1_weights;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.biases = nnwrkld_conv4_g1_biases;

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_4_g1_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_4_g1_conv->output_format.format_3d ={ { 13, 13, 384/2 } };
        }

        di->workflow_item_create_function(&wrkflwi_stage_4_g2_conv, 1, &wrkflwi_stage_3_4_g2_subv); // conv g2
        {
            wrkflwi_stage_4_g2_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_4_g2_conv->name = "c4g2";

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.weights = nnwrkld_conv4_g2_weights;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.biases = nnwrkld_conv4_g2_biases;

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_4_g2_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_4_g2_conv->output_format.format_3d ={ { 13, 13, 384/2 } };
        }


        // ------------------------------------------------------------------------------------------
        // STAGE 05
        //           convo: 3x3 stride 1x1; ReLU; 0-padded; output: 13x13x(2*256/2)
        //           merge: (z-axis)
        //         maxpool: 3x3 stride 2x2;
        //          output: 13x13x256
        di->workflow_item_create_function(&wrkflwi_stage_5_g1_conv, 1, &wrkflwi_stage_4_g1_conv); // conv g1
        {
            wrkflwi_stage_5_g1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_5_g1_conv->name = "c5g1";

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.weights = nnwrkld_conv5_g1_weights;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.biases = nnwrkld_conv5_g1_biases;

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_5_g1_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_5_g1_conv->output_format.format_3d ={ { 13, 13, 256/2 } };
        }

        di->workflow_item_create_function(&wrkflwi_stage_5_g2_conv, 1, &wrkflwi_stage_4_g2_conv); // conv g2
        {
            wrkflwi_stage_5_g2_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_5_g2_conv->name = "c5g2";

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.weights = nnwrkld_conv5_g2_weights;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.biases = nnwrkld_conv5_g2_biases;

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_5_g2_conv->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_5_g2_conv->output_format.format_3d ={ { 13, 13, 256/2 } };
        }
        // merge g1 and g2
        nn_workflow_item_t *input_merge_2[2] ={ wrkflwi_stage_5_g1_conv, wrkflwi_stage_5_g2_conv };
        di->workflow_item_create_function(&wrkflwi_stage_5_merge, 2, input_merge_2);
        {
            wrkflwi_stage_5_merge->type = NN_WORK_ITEM_TYPE_MERGE;
            wrkflwi_stage_5_merge->arguments.forward_merge.axis = 2; // value 2 for z-axis

            wrkflwi_stage_5_merge->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_5_merge->output_format.format_3d ={ { 13, 13, 256 } };
        }

        // maxpool: 3x3 stride 2x2;
        di->workflow_item_create_function(&wrkflwi_stage_5_pool, 1, &wrkflwi_stage_5_merge); // pooling
        {
            wrkflwi_stage_5_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_5_pool->name = "p5";

            wrkflwi_stage_5_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

            wrkflwi_stage_5_pool->arguments.forward_pooling.size[0] = 3;
            wrkflwi_stage_5_pool->arguments.forward_pooling.size[1] = 3;

            wrkflwi_stage_5_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_5_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_5_pool->output_format.format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_5_pool->output_format.format_3d ={ { 6, 6, 256 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 06
        //            full: ReLU
        //          output: 4096

        di->workflow_item_create_function(&wrkflwi_stage_6_fc, 1, &wrkflwi_stage_5_pool);
        {
            wrkflwi_stage_6_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_6_fc->name = "fc6";

            wrkflwi_stage_6_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_RELU;

            wrkflwi_stage_6_fc->arguments.forward_fully_connected.weights = nnwrkld_fc6_weights;
            wrkflwi_stage_6_fc->arguments.forward_fully_connected.biases = nnwrkld_fc6_biases;

            wrkflwi_stage_6_fc->output_format.format = NN_DATA_FORMAT_1D;
            wrkflwi_stage_6_fc->output_format.format_1d ={ { 4096 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 07
        //            full: ReLU
        //          output: 4096

        di->workflow_item_create_function(&wrkflwi_stage_7_fc, 1, &wrkflwi_stage_6_fc);
        {
            wrkflwi_stage_7_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_7_fc->name = "fc7";
            wrkflwi_stage_7_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_RELU;

            wrkflwi_stage_7_fc->arguments.forward_fully_connected.weights = nnwrkld_fc7_weights;
            wrkflwi_stage_7_fc->arguments.forward_fully_connected.biases = nnwrkld_fc7_biases;

            wrkflwi_stage_7_fc->output_format.format = NN_DATA_FORMAT_1D;
            wrkflwi_stage_7_fc->output_format.format_1d ={ { 4096 } };
        }
        // ------------------------------------------------------------------------------------------
        // STAGE 08
        //            full: ;
        //          output: 1000

        di->workflow_item_create_function(&wrkflwi_stage_8_fc, 1, &wrkflwi_stage_7_fc);
        {
            wrkflwi_stage_8_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_8_fc->name = "fc8";

            wrkflwi_stage_8_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            wrkflwi_stage_8_fc->arguments.forward_fully_connected.weights = nnwrkld_fc8_weights;
            wrkflwi_stage_8_fc->arguments.forward_fully_connected.biases = nnwrkld_fc8_biases;

            wrkflwi_stage_8_fc->output_format.format = NN_DATA_FORMAT_1D;
            wrkflwi_stage_8_fc->output_format.format_1d ={ { 1000 } };
        }
        // ------------------------------------------------------------------------------------------
        // STAGE 09 (softmax)
        //          output: 1000

        di->workflow_item_create_function(&wrkflwi_softmax, 1, &wrkflwi_stage_8_fc);
        {
            wrkflwi_softmax->type = NN_WORK_ITEM_TYPE_SOFTMAX;

            wrkflwi_softmax->output_format.format = NN_DATA_FORMAT_1D;
            wrkflwi_softmax->output_format.format_1d ={ { 1000 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 10 (output)
        //          output: 1000

        di->workflow_item_create_function(&wrkflwi_output, 1, &wrkflwi_softmax);
        {
            wrkflwi_output->type = NN_WORK_ITEM_TYPE_OUTPUT;

            wrkflwi_output->output_format.format = NN_DATA_FORMAT_1D;
            wrkflwi_output->output_format.format_1d ={ { 1000 } };

        }

        // -------------------------------------------------------------------------------------------
        // END of workflow stages definition
        // -------------------------------------------------------------------------------------------
        workflow->input[0] = wrkflwi_input;
        workflow->output[0] = wrkflwi_output;
        // -------------------------------------------------------------------------------------------

        return workflow;
    }
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran befor main execution starts.
// The sole function of this construction is attaching this workflow builder to
// library of workflow builders (singleton command pattern).
namespace {
    struct attach {
        workflow_builder_caffenet_float builder;
        attach() {
            workflow_builder::instance().add("caffenet_float", &builder);
        }
    };

    attach attach_;
}