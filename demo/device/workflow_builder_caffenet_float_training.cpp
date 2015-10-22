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
#include "common/nn_data_tools.h"

#include <algorithm>
#include <random>

const static uint32_t C_num_inputs = 5;
const static uint32_t C_num_outputs = 2;

const static nn::output_format input_format(227, 227, 3);
const static nn::output_format output_format(1000);
const static nn::output_format output_loss_format(1);

static NN_WORKLOAD_DATA_TYPE in_formats[C_num_inputs] =
        { NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH, NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_I32_1D, NN_WORKLOAD_DATA_TYPE_I32_1D, NN_WORKLOAD_DATA_TYPE_F32_1D };

static NN_WORKLOAD_DATA_TYPE out_formats[C_num_outputs] =
        { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_F32_1D };

class workflow_builder_caffenet_float_training: public workflow_builder_base
{

public:
    workflow_builder_caffenet_float_training() : workflow_builder_base(input_format.format_3d.size[0])
    {
        RGB_order = false;
        image_process = fi::resize_image_to_square;
    }

    bool is_valid() { return error_.empty(); }

    virtual NN_WORKLOAD_DATA_TYPE* get_input_formats() {return in_formats;}
    virtual NN_WORKLOAD_DATA_TYPE* get_output_formats() {return out_formats;}

private:

    void randomize_buffer(
        float mean,
        float std,
        nn::data<float>* &buffer,
        uint32_t seed)
    {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dis(mean, std);

        switch(buffer->dimension)
        {
        case 1:
            for (uint32_t x = 0; x < buffer->size[0]; ++x)
                (*buffer)(x) = dis(gen);
            break;
        case 2:
            for (uint32_t y = 0; y < buffer->size[1]; ++y)
                for (uint32_t x = 0; x < buffer->size[0]; ++x)
                    (*buffer)(x, y) = dis(gen);
            break;
        case 3:
            for (uint32_t z = 0; z < buffer->size[2]; ++z)
                for (uint32_t y = 0; y < buffer->size[1]; ++y)
                    for (uint32_t x = 0; x < buffer->size[0]; ++x)
                        (*buffer)(x, y, z) = dis(gen);
            break;
        case 4:
            for (uint32_t n = 0; n < buffer->size[3]; ++n)
                for (uint32_t z = 0; z < buffer->size[2]; ++z)
                    for (uint32_t y = 0; y < buffer->size[1]; ++y)
                        for (uint32_t x = 0; x < buffer->size[0]; ++x)
                            (*buffer)(x, y, z, n) = dis(gen);
            break;
        default:
            assert(0);
            break;
        }
    }

    void set_buffer(
        float value,
        nn::data<float>* &buffer)
    {
        switch(buffer->dimension)
        {
        case 1:
            for (uint32_t x = 0; x < buffer->size[0]; ++x)
                (*buffer)(x) = value;
            break;
        case 2:
            for (uint32_t y = 0; y < buffer->size[1]; ++y)
                for (uint32_t x = 0; x < buffer->size[0]; ++x)
                    (*buffer)(x, y) = value;
            break;
        case 3:
            for (uint32_t z = 0; z < buffer->size[2]; ++z)
                for (uint32_t y = 0; y < buffer->size[1]; ++y)
                    for (uint32_t x = 0; x < buffer->size[0]; ++x)
                        (*buffer)(x, y, z) = value;
            break;
        case 4:
            for (uint32_t n = 0; n < buffer->size[3]; ++n)
                for (uint32_t z = 0; z < buffer->size[2]; ++z)
                    for (uint32_t y = 0; y < buffer->size[1]; ++y)
                        for (uint32_t x = 0; x < buffer->size[0]; ++x)
                            (*buffer)(x, y, z, n) = value;
            break;
        default:
            assert(0);
            break;
        }
    }

    std::string error_;

    // pointers to successive workflow parts
    nn_workflow_item_t
        *wrkflwi_forward_input,
        *wrkflwi_dropout_seed,
        *wrkflwi_execution_mode,
        *wrkflwi_learning_rate,
        *wrkflwi_stage_0_mean_substract,
        *wrkflwi_stage_1_conv,
        *wrkflwi_stage_1_conv_relu,
        *wrkflwi_stage_1_pool,
        *wrkflwi_stage_1_norm,
        *wrkflwi_stage_1_2_g1_subv,
        *wrkflwi_stage_1_2_g2_subv,
        *wrkflwi_stage_2_g1_conv,
        *wrkflwi_stage_2_g2_conv,
        *wrkflwi_stage_2_merge,
        *wrkflwi_stage_2_conv_relu,
        *wrkflwi_stage_2_pool,
        *wrkflwi_stage_2_norm,
        *wrkflwi_stage_3_conv,
        *wrkflwi_stage_3_conv_relu,
        *wrkflwi_stage_3_4_g1_subv,
        *wrkflwi_stage_3_4_g2_subv,
        *wrkflwi_stage_4_g1_conv,
        *wrkflwi_stage_4_g1_conv_relu,
        *wrkflwi_stage_4_g2_conv,
        *wrkflwi_stage_4_g2_conv_relu,
        *wrkflwi_stage_5_g1_conv,
        *wrkflwi_stage_5_g2_conv,
        *wrkflwi_stage_5_merge,
        *wrkflwi_stage_5_conv_relu,
        *wrkflwi_stage_5_pool,
        *wrkflwi_stage_6_fc,
        *wrkflwi_stage_6_fc_relu,
        *wrkflwi_stage_6_fc_dropout,
        *wrkflwi_stage_7_fc,
        *wrkflwi_stage_7_fc_relu,
        *wrkflwi_stage_7_fc_dropout,
        *wrkflwi_stage_8_fc,
        *wrkflwi_softmax,
        *wrkflwi_softmax_loss,
        *wrkflwi_output,
        *wrkflwi_output_loss,
        *wrkflwi_backward_input,
        *wrkflwi_softmax_loss_backprop,
        *wrkflwi_stage_8_fc_backprop,
        *wrkflwi_stage_8_fc_update_backprop,
        *wrkflwi_stage_7_fc_dropout_backprop,
        *wrkflwi_stage_7_fc_relu_backprop,
        *wrkflwi_stage_7_fc_backprop,
        *wrkflwi_stage_7_fc_update_backprop,
        *wrkflwi_stage_6_fc_dropout_backprop,
        *wrkflwi_stage_6_fc_relu_backprop,
        *wrkflwi_stage_6_fc_backprop,
        *wrkflwi_stage_6_fc_update_backprop,
        *wrkflwi_stage_5_pool_backprop,
        *wrkflwi_stage_5_conv_relu_backprop,
        *wrkflwi_stage_5_g1_subv_backprop,
        *wrkflwi_stage_5_g2_subv_backprop,
        *wrkflwi_stage_5_g1_conv_backprop,
        *wrkflwi_stage_5_g1_conv_update_backprop,
        *wrkflwi_stage_5_g2_conv_backprop,
        *wrkflwi_stage_5_g2_conv_update_backprop,
        *wrkflwi_stage_4_g1_conv_relu_backprop,
        *wrkflwi_stage_4_g2_conv_relu_backprop,
        *wrkflwi_stage_4_g1_conv_backprop,
        *wrkflwi_stage_4_g1_conv_update_backprop,
        *wrkflwi_stage_4_g2_conv_backprop,
        *wrkflwi_stage_4_g2_conv_update_backprop,
        *wrkflwi_stage_4_merge_backprop,
        *wrkflwi_stage_3_conv_relu_backprop,
        *wrkflwi_stage_3_conv_backprop,
        *wrkflwi_stage_3_conv_update_backprop,
        *wrkflwi_stage_2_norm_backprop,
        *wrkflwi_stage_2_pool_backprop,
        *wrkflwi_stage_2_conv_relu_backprop,
        *wrkflwi_stage_2_g1_subv_backprop,
        *wrkflwi_stage_2_g2_subv_backprop,
        *wrkflwi_stage_2_g1_conv_backprop,
        *wrkflwi_stage_2_g1_conv_update_backprop,
        *wrkflwi_stage_2_g2_conv_backprop,
        *wrkflwi_stage_2_g2_conv_update_backprop,
        *wrkflwi_stage_2_merge_backprop,
        *wrkflwi_stage_1_norm_backprop,
        *wrkflwi_stage_1_pool_backprop,
        *wrkflwi_stage_1_conv_relu_backprop,
        *wrkflwi_stage_1_conv_backprop,
        *wrkflwi_stage_1_conv_update_backprop
        ;

    // pointers to <nn_workload_data>s containing weights and biases;
    nn::data<float>
        *nnwrkld_imagenet_mean      = nullptr,
        *nnwrkld_conv1_weights      = nullptr,
        *nnwrkld_conv1_biases       = nullptr,
        *nnwrkld_conv2_g1_weights   = nullptr,
        *nnwrkld_conv2_g1_biases    = nullptr,
        *nnwrkld_conv2_g2_weights   = nullptr,
        *nnwrkld_conv2_g2_biases    = nullptr,
        *nnwrkld_conv3_weights      = nullptr,
        *nnwrkld_conv3_biases       = nullptr,
        *nnwrkld_conv4_g1_weights   = nullptr,
        *nnwrkld_conv4_g1_biases    = nullptr,
        *nnwrkld_conv4_g2_weights   = nullptr,
        *nnwrkld_conv4_g2_biases    = nullptr,
        *nnwrkld_conv5_g1_weights   = nullptr,
        *nnwrkld_conv5_g1_biases    = nullptr,
        *nnwrkld_conv5_g2_weights   = nullptr,
        *nnwrkld_conv5_g2_biases    = nullptr,
        *nnwrkld_fc6_weights        = nullptr,
        *nnwrkld_fc6_biases         = nullptr,
        *nnwrkld_fc7_weights        = nullptr,
        *nnwrkld_fc7_biases         = nullptr,
        *nnwrkld_fc8_weights        = nullptr,
        *nnwrkld_fc8_biases         = nullptr;

    nn_workflow_t           *workflow;
    nn_device_interface_0_t *di;

public:

    void cleanup()
    {
        if(!is_valid()) throw std::runtime_error(error_);

        /* ****************************************************************************************** */
        /* Cleanup in memory                                                                          */
        /* ****************************************************************************************** */
        std::cout
            << "Cleanup in memory"
            << std::endl
            << "========================================================"
            << std::endl;

        di->workflow_item_delete_function(wrkflwi_forward_input);
        di->workflow_item_delete_function(wrkflwi_dropout_seed);
        di->workflow_item_delete_function(wrkflwi_execution_mode);
        di->workflow_item_delete_function(wrkflwi_learning_rate);
        di->workflow_item_delete_function(wrkflwi_stage_0_mean_substract);
        di->workflow_item_delete_function(wrkflwi_stage_1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_1_conv_relu);
        di->workflow_item_delete_function(wrkflwi_stage_1_pool);
        di->workflow_item_delete_function(wrkflwi_stage_1_norm);
        di->workflow_item_delete_function(wrkflwi_stage_1_2_g1_subv);
        di->workflow_item_delete_function(wrkflwi_stage_1_2_g2_subv);
        di->workflow_item_delete_function(wrkflwi_stage_2_g1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_2_g2_conv);
        di->workflow_item_delete_function(wrkflwi_stage_2_merge);
        di->workflow_item_delete_function(wrkflwi_stage_2_conv_relu);
        di->workflow_item_delete_function(wrkflwi_stage_2_pool);
        di->workflow_item_delete_function(wrkflwi_stage_2_norm);
        di->workflow_item_delete_function(wrkflwi_stage_3_conv);
        di->workflow_item_delete_function(wrkflwi_stage_3_conv_relu);
        di->workflow_item_delete_function(wrkflwi_stage_3_4_g1_subv);
        di->workflow_item_delete_function(wrkflwi_stage_3_4_g2_subv);
        di->workflow_item_delete_function(wrkflwi_stage_4_g1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_4_g1_conv_relu);
        di->workflow_item_delete_function(wrkflwi_stage_4_g2_conv);
        di->workflow_item_delete_function(wrkflwi_stage_4_g2_conv_relu);
        di->workflow_item_delete_function(wrkflwi_stage_5_g1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_5_g2_conv);
        di->workflow_item_delete_function(wrkflwi_stage_5_merge);
        di->workflow_item_delete_function(wrkflwi_stage_5_conv_relu);
        di->workflow_item_delete_function(wrkflwi_stage_5_pool);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc_relu);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc_dropout);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc_relu);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc_dropout);
        di->workflow_item_delete_function(wrkflwi_stage_8_fc);
        di->workflow_item_delete_function(wrkflwi_softmax);
        di->workflow_item_delete_function(wrkflwi_softmax_loss);
        di->workflow_item_delete_function(wrkflwi_output);
        di->workflow_item_delete_function(wrkflwi_output_loss);
        di->workflow_item_delete_function(wrkflwi_backward_input);
        di->workflow_item_delete_function(wrkflwi_softmax_loss_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_8_fc_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_8_fc_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc_dropout_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_7_fc_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc_dropout_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_6_fc_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_pool_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_conv_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_g1_subv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_g2_subv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_g1_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_g1_conv_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_g2_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_5_g2_conv_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_4_g1_conv_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_4_g2_conv_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_4_g1_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_4_g1_conv_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_4_g2_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_4_g2_conv_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_4_merge_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_3_conv_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_3_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_3_conv_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_norm_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_pool_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_conv_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_g1_subv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_g2_subv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_g1_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_g1_conv_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_g2_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_g2_conv_update_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_2_merge_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_1_norm_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_1_pool_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_1_conv_relu_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_1_conv_backprop);
        di->workflow_item_delete_function(wrkflwi_stage_1_conv_update_backprop);

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

    virtual nn_workflow_t *init_workflow(nn_device_interface_0_t *di)
    {

        if(!is_valid()) throw std::runtime_error(error_);

        this->di = di;

        std::cout
            << "--------------------------------------------------------"
            << std::endl
            << "Loading weights and biases"
            << std::endl << std::endl;

        nnwrkld_imagenet_mean  = nn_data_load_from_file("weights_caffenet/imagenet_mean.nnd");

        // Check for previously saved parameters.
        if(    !!(nnwrkld_conv1_weights    = nn_data_load_from_file("weights_caffenet_training/conv1_weights.nnd"    ))
            && !!(nnwrkld_conv1_biases     = nn_data_load_from_file("weights_caffenet_training/conv1_biases.nnd"     ))
            && !!(nnwrkld_conv2_g1_weights = nn_data_load_from_file("weights_caffenet_training/conv2_g1_weights.nnd" ))
            && !!(nnwrkld_conv2_g1_biases  = nn_data_load_from_file("weights_caffenet_training/conv2_g1_biases.nnd"  ))
            && !!(nnwrkld_conv2_g2_weights = nn_data_load_from_file("weights_caffenet_training/conv2_g2_weights.nnd" ))
            && !!(nnwrkld_conv2_g2_biases  = nn_data_load_from_file("weights_caffenet_training/conv2_g2_biases.nnd"  ))
            && !!(nnwrkld_conv3_weights    = nn_data_load_from_file("weights_caffenet_training/conv3_weights.nnd"    ))
            && !!(nnwrkld_conv3_biases     = nn_data_load_from_file("weights_caffenet_training/conv3_biases.nnd"     ))
            && !!(nnwrkld_conv4_g1_weights = nn_data_load_from_file("weights_caffenet_training/conv4_g1_weights.nnd" ))
            && !!(nnwrkld_conv4_g1_biases  = nn_data_load_from_file("weights_caffenet_training/conv4_g1_biases.nnd"  ))
            && !!(nnwrkld_conv4_g2_weights = nn_data_load_from_file("weights_caffenet_training/conv4_g2_weights.nnd" ))
            && !!(nnwrkld_conv4_g2_biases  = nn_data_load_from_file("weights_caffenet_training/conv4_g2_biases.nnd"  ))
            && !!(nnwrkld_conv5_g1_weights = nn_data_load_from_file("weights_caffenet_training/conv5_g1_weights.nnd" ))
            && !!(nnwrkld_conv5_g1_biases  = nn_data_load_from_file("weights_caffenet_training/conv5_g1_biases.nnd"  ))
            && !!(nnwrkld_conv5_g2_weights = nn_data_load_from_file("weights_caffenet_training/conv5_g2_weights.nnd" ))
            && !!(nnwrkld_conv5_g2_biases  = nn_data_load_from_file("weights_caffenet_training/conv5_g2_biases.nnd"  ))
            && !!(nnwrkld_fc6_weights      = nn_data_load_from_file("weights_caffenet_training/fc6_weights.nnd"      ))
            && !!(nnwrkld_fc6_biases       = nn_data_load_from_file("weights_caffenet_training/fc6_biases.nnd"       ))
            && !!(nnwrkld_fc7_weights      = nn_data_load_from_file("weights_caffenet_training/fc7_weights.nnd"      ))
            && !!(nnwrkld_fc7_biases       = nn_data_load_from_file("weights_caffenet_training/fc7_biases.nnd"       ))
            && !!(nnwrkld_fc8_weights      = nn_data_load_from_file("weights_caffenet_training/fc8_weights.nnd"      ))
            && !!(nnwrkld_fc8_biases       = nn_data_load_from_file("weights_caffenet_training/fc8_biases.nnd"       )))
        {
            std::cout << "Using previous iteration buffers." << std::endl;
        }
        else
        {   // If any param file wasn't found - setup all initial parameters.
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

            nnwrkld_conv1_weights      = new nn::data<float>(11, 11, 3, 96);
            nnwrkld_conv1_biases       = new nn::data<float>(96);
            nnwrkld_conv2_g1_weights   = new nn::data<float>(5, 5, 48, 128);
            nnwrkld_conv2_g1_biases    = new nn::data<float>(128);
            nnwrkld_conv2_g2_weights   = new nn::data<float>(5, 5, 48, 128);
            nnwrkld_conv2_g2_biases    = new nn::data<float>(128);
            nnwrkld_conv3_weights      = new nn::data<float>(3, 3, 256, 384);
            nnwrkld_conv3_biases       = new nn::data<float>(384);
            nnwrkld_conv4_g1_weights   = new nn::data<float>(3, 3, 192, 192);
            nnwrkld_conv4_g1_biases    = new nn::data<float>(192);
            nnwrkld_conv4_g2_weights   = new nn::data<float>(3, 3, 192, 192);
            nnwrkld_conv4_g2_biases    = new nn::data<float>(192);
            nnwrkld_conv5_g1_weights   = new nn::data<float>(3, 3, 192, 128);
            nnwrkld_conv5_g1_biases    = new nn::data<float>(128);
            nnwrkld_conv5_g2_weights   = new nn::data<float>(3, 3, 192, 128);
            nnwrkld_conv5_g2_biases    = new nn::data<float>(128);
            nnwrkld_fc6_weights        = new nn::data<float>(6, 6, 256, 4096);
            nnwrkld_fc6_biases         = new nn::data<float>(4096);
            nnwrkld_fc7_weights        = new nn::data<float>(4096, 4096);
            nnwrkld_fc7_biases         = new nn::data<float>(4096);
            nnwrkld_fc8_weights        = new nn::data<float>(4096,1000);
            nnwrkld_fc8_biases         = new nn::data<float>(1000);

            randomize_buffer(0.0f, 0.01f, nnwrkld_conv1_weights, 1);
            set_buffer(0.0f, nnwrkld_conv1_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_conv2_g1_weights, 2);
            set_buffer(1.0f, nnwrkld_conv2_g1_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_conv2_g2_weights, 3);
            set_buffer(1.0f, nnwrkld_conv2_g2_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_conv3_weights, 4);
            set_buffer(0.0f, nnwrkld_conv3_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_conv4_g1_weights, 5);
            set_buffer(1.0f, nnwrkld_conv4_g1_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_conv4_g2_weights, 6);
            set_buffer(1.0f, nnwrkld_conv4_g2_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_conv5_g1_weights, 7);
            set_buffer(1.0f, nnwrkld_conv5_g1_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_conv5_g2_weights, 8);
            set_buffer(1.0f, nnwrkld_conv5_g2_biases);
            randomize_buffer(0.0f, 0.005f, nnwrkld_fc6_weights, 9);
            set_buffer(1.0f, nnwrkld_fc6_biases);
            randomize_buffer(0.0f, 0.005f, nnwrkld_fc7_weights, 10);
            set_buffer(1.0f, nnwrkld_fc7_biases);
            randomize_buffer(0.0f, 0.01f, nnwrkld_fc8_weights, 11);
            set_buffer(0.0f, nnwrkld_fc8_biases);

            std::cout << "Using newly initialized buffers." << std::endl;
        }

        std::cout
            << "--------------------------------------------------------" << std::endl
            << "Build of workflow" << std::endl;

        di->workflow_create_function(&workflow, C_num_inputs, C_num_outputs);

        // ------------------------------------------------------------------------------------------
        // STAGE 0 (input)
        //         output: 227x227x3
        {
            di->workflow_item_create_function(&wrkflwi_forward_input, 0, nullptr, 1);

            wrkflwi_forward_input->type = NN_WORK_ITEM_TYPE_INPUT;
            wrkflwi_forward_input->arguments.input.index = 0;
            wrkflwi_forward_input->output_format[0] = input_format;
        }

        {
            di->workflow_item_create_function(&wrkflwi_dropout_seed, 0, nullptr, 1);

            wrkflwi_dropout_seed->type = NN_WORK_ITEM_TYPE_INPUT;
            wrkflwi_dropout_seed->arguments.input.index = 2;
            wrkflwi_dropout_seed->output_format[0] = nn::output_format{1};
        }

        {
            di->workflow_item_create_function(&wrkflwi_execution_mode, 0, nullptr, 1);

            wrkflwi_execution_mode->type = NN_WORK_ITEM_TYPE_INPUT;
            wrkflwi_execution_mode->arguments.input.index = 3;
            wrkflwi_execution_mode->output_format[0] = nn::output_format{1};
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 0 (imagenet_mean_subtract)
        //         output: 227x227x3
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_forward_input, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_0_mean_substract, 1, &inputs_descriptor, 1);

            wrkflwi_stage_0_mean_substract->type = NN_WORK_ITEM_TYPE_ARITHMETIC;
            wrkflwi_stage_0_mean_substract->arguments.forward_arithmetic.factor = nnwrkld_imagenet_mean;
            wrkflwi_stage_0_mean_substract->arguments.forward_arithmetic.arithmetic_function = NN_ARITHMETIC_FUNCTION_SUBTRACTION;

            wrkflwi_stage_0_mean_substract->output_format[0] = input_format;
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 01
        //           convo: 11x11 stride 4x4; ReLU; output: 55x55x96
        //         maxpool: 3x3 stride 2x2;
        //            norm: RESPONSE_ACROSS_MAPS
        //          output: 27x27x96
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_0_mean_substract, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_conv, 1, &inputs_descriptor, 1);

            wrkflwi_stage_1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_1_conv->name = "conv1";

            wrkflwi_stage_1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            wrkflwi_stage_1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            wrkflwi_stage_1_conv->arguments.forward_convolution.weights = nnwrkld_conv1_weights;
            wrkflwi_stage_1_conv->arguments.forward_convolution.biases = nnwrkld_conv1_biases;

            wrkflwi_stage_1_conv->arguments.forward_convolution.center_offset[0] = 0;
            wrkflwi_stage_1_conv->arguments.forward_convolution.center_offset[1] = 0;

            wrkflwi_stage_1_conv->arguments.forward_convolution.stride[0] = 4;
            wrkflwi_stage_1_conv->arguments.forward_convolution.stride[1] = 4;

            wrkflwi_stage_1_conv->output_format[0] = nn::output_format{ 55, 55, 96 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_conv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_conv_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_1_conv_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_1_conv_relu->name = "relu1";

            wrkflwi_stage_1_conv_relu->output_format[0] = nn::output_format{ 55, 55, 96 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_conv_relu, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_pool, 1, &inputs_descriptor, 2);

            wrkflwi_stage_1_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_1_pool->name = "p1";

            wrkflwi_stage_1_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            wrkflwi_stage_1_pool->arguments.forward_pooling.size[0] = 3;
            wrkflwi_stage_1_pool->arguments.forward_pooling.size[1] = 3;
            wrkflwi_stage_1_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_1_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_1_pool->output_format[0] = nn::output_format{ 27, 27, 96 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_pool, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_norm, 1, &inputs_descriptor, 2);

            wrkflwi_stage_1_norm->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            wrkflwi_stage_1_norm->name = "lrn1";

            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.k = 1; // in Krishevsky's article is 2
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.n = 5;
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.alpha = 0.0001f/5; // in Krishevsky's paper is 1e-4,
                                                                                                    // but didn't write that sum of the squares
                                                                                                    // is divided by number of elements (n)
            wrkflwi_stage_1_norm->arguments.forward_normalization.normalization.beta = 0.75f;

            wrkflwi_stage_1_norm->output_format[0] = nn::output_format{ 27, 27, 96 };

            // Intermediate data.
            wrkflwi_stage_1_norm->output_format[1] = nn::output_format{ 27, 27, 96 };
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
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_norm, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_2_g1_subv, 1, &inputs_descriptor, 1); // view g1

            wrkflwi_stage_1_2_g1_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_1_2_g1_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_1_2_g1_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_1_2_g1_subv->arguments.view.origin[2] = 0;

            wrkflwi_stage_1_2_g1_subv->output_format[0] = nn::output_format{ 27, 27, 96/2 };

        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_norm, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_2_g2_subv, 1, &inputs_descriptor, 1);   // view g2
            wrkflwi_stage_1_2_g2_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_1_2_g2_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_1_2_g2_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_1_2_g2_subv->arguments.view.origin[2] = (96/2);

            wrkflwi_stage_1_2_g2_subv->output_format[0] = nn::output_format{ 27, 27, 96/2 };
        }

        // convolution 2, g1: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_2_g1_subv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_g1_conv, 1, &inputs_descriptor, 1);

            wrkflwi_stage_2_g1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_2_g1_conv->name = "conv2_g1";

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.weights = nnwrkld_conv2_g1_weights;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.biases = nnwrkld_conv2_g1_biases;

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.center_offset[0] = 2;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.center_offset[1] = 2;

            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_2_g1_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_2_g1_conv->output_format[0] = nn::output_format{ 27, 27, 256/2 };
        }

        // convolution 2, g2: 5x5 stride 1x1; ReLU; 0-padded output: 13x13x(2*96/2)
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_2_g2_subv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_g2_conv, 1, &inputs_descriptor, 1);

            wrkflwi_stage_2_g2_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_2_g2_conv->name = "conv2_g2";

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.weights = nnwrkld_conv2_g2_weights;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.biases = nnwrkld_conv2_g2_biases;

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.center_offset[0] = 2;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.center_offset[1] = 2;

            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_2_g2_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_2_g2_conv->output_format[0] = nn::output_format{ 27, 27, 256/2 };
        }

        // merge g1 and g2
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_2_g1_conv, 0 }, { wrkflwi_stage_2_g2_conv, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_2_merge, 2, inputs_descriptor, 1);

            wrkflwi_stage_2_merge->type = NN_WORK_ITEM_TYPE_MERGE;
            wrkflwi_stage_2_merge->arguments.forward_merge.axis = 2; // value 2 for z-axis

            wrkflwi_stage_2_merge->output_format[0] = nn::output_format{ 27, 27, 256 };

        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_merge, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_conv_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_2_conv_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_2_conv_relu->name = "relu2";

            wrkflwi_stage_2_conv_relu->output_format[0] = nn::output_format{ 27, 27, 256 };
        }

        // maxpool: 3x3 stride 2x2;
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_conv_relu, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_pool, 1, &inputs_descriptor, 2); // pooling

            wrkflwi_stage_2_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_2_pool->name = "p2";

            wrkflwi_stage_2_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

            wrkflwi_stage_2_pool->arguments.forward_pooling.size[0] = 3;
            wrkflwi_stage_2_pool->arguments.forward_pooling.size[1] = 3;

            wrkflwi_stage_2_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_2_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_2_pool->output_format[0] = nn::output_format{ 13, 13, 256 };
        }

        //norm: RESPONSE_ACROSS_MAPS; output: 13x13x256
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_pool, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_norm, 1, &inputs_descriptor, 2);

            wrkflwi_stage_2_norm->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            wrkflwi_stage_2_norm->name = "lrn2";

            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.k = 1;              // |
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.n = 5;              // |
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.alpha = 0.0001f/5;  // > see coment at wrkflwi_stage_1_norm
            wrkflwi_stage_2_norm->arguments.forward_normalization.normalization.beta = 0.75f;       // |

            wrkflwi_stage_2_norm->output_format[0] = nn::output_format{ 13, 13, 256 };

            // Intermediate data.
            wrkflwi_stage_2_norm->output_format[1] = nn::output_format{ 13, 13, 256 };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 03
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x384
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_norm, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_3_conv, 1, &inputs_descriptor, 1);

            wrkflwi_stage_3_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_3_conv->name = "conv3";
            wrkflwi_stage_3_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_3_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_3_conv->arguments.forward_convolution.weights = nnwrkld_conv3_weights;
            wrkflwi_stage_3_conv->arguments.forward_convolution.biases = nnwrkld_conv3_biases;

            wrkflwi_stage_3_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_3_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_3_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_3_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_3_conv->output_format[0] = nn::output_format{ 13, 13, 384 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_3_conv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_3_conv_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_3_conv_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_3_conv_relu->name = "relu3";

            wrkflwi_stage_3_conv_relu->output_format[0] = nn::output_format{ 13, 13, 384 };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 04
        //           split: 2 (z-axis 384/2)
        //           convo: 3x3 stride 1x1; ReLU; 0-padded
        //          output: 13x13x(2*384/2) (continue split to next stage)
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_3_conv_relu, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_3_4_g1_subv, 1, &inputs_descriptor, 1); // view g1

            wrkflwi_stage_3_4_g1_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_3_4_g1_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_3_4_g1_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_3_4_g1_subv->arguments.view.origin[2] = 0;

            wrkflwi_stage_3_4_g1_subv->output_format[0] = nn::output_format{ 13, 13, 384/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_3_conv_relu, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_3_4_g2_subv, 1, &inputs_descriptor, 1); // view g2

            wrkflwi_stage_3_4_g2_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_3_4_g2_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_3_4_g2_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_3_4_g2_subv->arguments.view.origin[2] = 384/2;

            wrkflwi_stage_3_4_g2_subv->output_format[0] = nn::output_format{ 13, 13, 384/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_3_4_g1_subv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g1_conv, 1, &inputs_descriptor, 1); // conv g1

            wrkflwi_stage_4_g1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_4_g1_conv->name = "conv4_g1";

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.weights = nnwrkld_conv4_g1_weights;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.biases = nnwrkld_conv4_g1_biases;

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_4_g1_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_4_g1_conv->output_format[0] = nn::output_format{ 13, 13, 384/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_g1_conv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g1_conv_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_4_g1_conv_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_4_g1_conv_relu->name = "relu4g1";

            wrkflwi_stage_4_g1_conv_relu->output_format[0] = nn::output_format{ 13, 13, 384/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_3_4_g2_subv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g2_conv, 1, &inputs_descriptor, 1); // conv g2

            wrkflwi_stage_4_g2_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_4_g2_conv->name = "conv4_g2";

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.weights = nnwrkld_conv4_g2_weights;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.biases = nnwrkld_conv4_g2_biases;

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_4_g2_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_4_g2_conv->output_format[0] = nn::output_format{ 13, 13, 384/2 };
        }


        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_g2_conv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g2_conv_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_4_g2_conv_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_4_g2_conv_relu->name = "relu4g2";

            wrkflwi_stage_4_g2_conv_relu->output_format[0] = nn::output_format{ 13, 13, 384/2 };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 05
        //           convo: 3x3 stride 1x1; ReLU; 0-padded; output: 13x13x(2*256/2)
        //           merge: (z-axis)
        //         maxpool: 3x3 stride 2x2;
        //          output: 13x13x256
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_g1_conv_relu, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_g1_conv, 1, &inputs_descriptor, 1); // conv g1

            wrkflwi_stage_5_g1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_5_g1_conv->name = "conv5_g1";

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.weights = nnwrkld_conv5_g1_weights;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.biases = nnwrkld_conv5_g1_biases;

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_5_g1_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_5_g1_conv->output_format[0] = nn::output_format{ 13, 13, 256/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_g2_conv_relu, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_g2_conv, 1, &inputs_descriptor, 1); // conv g2

            wrkflwi_stage_5_g2_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_5_g2_conv->name = "conv5_g2";

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.weights = nnwrkld_conv5_g2_weights;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.biases = nnwrkld_conv5_g2_biases;

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.center_offset[0] = 1;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.center_offset[1] = 1;

            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_5_g2_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_5_g2_conv->output_format[0] = nn::output_format{ 13, 13, 256/2 };
        }

        // merge g1 and g2
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_5_g1_conv, 0 }, { wrkflwi_stage_5_g2_conv, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_5_merge, 2, inputs_descriptor, 1);

            wrkflwi_stage_5_merge->type = NN_WORK_ITEM_TYPE_MERGE;
            wrkflwi_stage_5_merge->arguments.forward_merge.axis = 2; // value 2 for z-axis

            wrkflwi_stage_5_merge->output_format[0] = nn::output_format{ 13, 13, 256 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_merge, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_conv_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_5_conv_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_5_conv_relu->name = "relu5";

            wrkflwi_stage_5_conv_relu->output_format[0] = nn::output_format{ 13, 13, 256 };
        }

        // maxpool: 3x3 stride 2x2;
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_conv_relu, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_pool, 1, &inputs_descriptor, 2); // pooling

            wrkflwi_stage_5_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_5_pool->name = "p5";

            wrkflwi_stage_5_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

            wrkflwi_stage_5_pool->arguments.forward_pooling.size[0] = 3;
            wrkflwi_stage_5_pool->arguments.forward_pooling.size[1] = 3;

            wrkflwi_stage_5_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_5_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_5_pool->output_format[0] = nn::output_format{ 6, 6, 256 };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 06
        //            full: ReLU
        //          output: 4096
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_pool, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_6_fc, 1, &inputs_descriptor, 1);

            wrkflwi_stage_6_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_6_fc->name = "fc6";

            wrkflwi_stage_6_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            wrkflwi_stage_6_fc->arguments.forward_fully_connected.weights = nnwrkld_fc6_weights;
            wrkflwi_stage_6_fc->arguments.forward_fully_connected.biases = nnwrkld_fc6_biases;

            wrkflwi_stage_6_fc->output_format[0] = nn::output_format{ 4096 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_6_fc, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_6_fc_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_6_fc_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_6_fc_relu->name = "relu6";

            wrkflwi_stage_6_fc_relu->output_format[0] = nn::output_format{ 4096 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_6_fc_relu, 0 }, { wrkflwi_dropout_seed, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_6_fc_dropout, 3, inputs_descriptor, 1);

            wrkflwi_stage_6_fc_dropout->type = NN_WORK_ITEM_TYPE_DROPOUT;
            wrkflwi_stage_6_fc_dropout->name = "dropout6";

            wrkflwi_stage_6_fc_dropout->arguments.dropout.drop_rate = 0.5f;

            wrkflwi_stage_6_fc_dropout->output_format[0] = nn::output_format{ 4096 };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 07
        //            full: ReLU
        //          output: 4096
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_6_fc_dropout, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_7_fc, 1, &inputs_descriptor, 1);

            wrkflwi_stage_7_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_7_fc->name = "fc7";
            wrkflwi_stage_7_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            wrkflwi_stage_7_fc->arguments.forward_fully_connected.weights = nnwrkld_fc7_weights;
            wrkflwi_stage_7_fc->arguments.forward_fully_connected.biases = nnwrkld_fc7_biases;

            wrkflwi_stage_7_fc->output_format[0] = nn::output_format{ 4096 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_7_fc, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_7_fc_relu, 1, &inputs_descriptor, 1);

            wrkflwi_stage_7_fc_relu->type = NN_WORK_ITEM_TYPE_RELU;
            wrkflwi_stage_7_fc_relu->name = "relu7";

            wrkflwi_stage_7_fc_relu->output_format[0] = nn::output_format{ 4096 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_7_fc_relu, 0 }, { wrkflwi_dropout_seed, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_7_fc_dropout, 3, inputs_descriptor, 1);

            wrkflwi_stage_7_fc_dropout->type = NN_WORK_ITEM_TYPE_DROPOUT;
            wrkflwi_stage_7_fc_dropout->name = "dropout7";

            wrkflwi_stage_7_fc_dropout->arguments.dropout.drop_rate = 0.5f;

            wrkflwi_stage_7_fc_dropout->output_format[0] = nn::output_format{ 4096 };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 08
        //            full: ;
        //          output: 1000
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_7_fc_dropout, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_8_fc, 1, &inputs_descriptor, 1);

            wrkflwi_stage_8_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_8_fc->name = "fc8";

            wrkflwi_stage_8_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            wrkflwi_stage_8_fc->arguments.forward_fully_connected.weights = nnwrkld_fc8_weights;
            wrkflwi_stage_8_fc->arguments.forward_fully_connected.biases = nnwrkld_fc8_biases;

            wrkflwi_stage_8_fc->output_format[0] = output_format;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_8_fc, 0 };
            di->workflow_item_create_function(&wrkflwi_softmax, 1, &inputs_descriptor, 1);
            wrkflwi_softmax->name = "softmax";
            wrkflwi_softmax->type = NN_WORK_ITEM_TYPE_SOFTMAX;

            wrkflwi_softmax->output_format[0] = output_format;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_softmax, 0 };
            di->workflow_item_create_function(&wrkflwi_output, 1, &inputs_descriptor, 1);
            wrkflwi_output->type = NN_WORK_ITEM_TYPE_OUTPUT;
            wrkflwi_output->name = "output";
            wrkflwi_output->arguments.output.index = 0;
            wrkflwi_output->output_format[0] = output_format;

        }

        // -------------------------------------------------------------------------------------------
        // END of forward workflow stages definition
        // -------------------------------------------------------------------------------------------
        workflow->input[0] = wrkflwi_forward_input;
        workflow->input[2] = wrkflwi_dropout_seed;
        workflow->input[3] = wrkflwi_execution_mode;
        workflow->output[0] = wrkflwi_output;

        // -------------------------------------------------------------------------------------------
        // START of backward workflow stages definition
        // -------------------------------------------------------------------------------------------
        {
            di->workflow_item_create_function(&wrkflwi_backward_input, 0, nullptr, 1);
            wrkflwi_backward_input->name = "input_target";
            wrkflwi_backward_input->type = NN_WORK_ITEM_TYPE_INPUT;
            wrkflwi_backward_input->arguments.input.index = 1;
            wrkflwi_backward_input->output_format[0] = nn::output_format{ 1 };
        }

        {
            di->workflow_item_create_function(&wrkflwi_learning_rate, 0, nullptr, 1);

            wrkflwi_learning_rate->type = NN_WORK_ITEM_TYPE_INPUT;
            wrkflwi_learning_rate->arguments.input.index = 4;
            wrkflwi_learning_rate->output_format[0] = nn::output_format{1};
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_8_fc, 0 }, { wrkflwi_backward_input, 0 } };
            di->workflow_item_create_function(&wrkflwi_softmax_loss, 2, inputs_descriptor, 2);
            wrkflwi_softmax_loss->name = "softmax_loss";
            wrkflwi_softmax_loss->type = NN_WORK_ITEM_TYPE_SOFTMAX_LOSS;

            wrkflwi_softmax_loss->output_format[0] = output_format;
            wrkflwi_softmax_loss->output_format[1] = output_loss_format;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_softmax_loss, 1 };
            di->workflow_item_create_function(&wrkflwi_output_loss, 1, &inputs_descriptor, 1);
            wrkflwi_output_loss->type = NN_WORK_ITEM_TYPE_OUTPUT;
            wrkflwi_output_loss->name = "output_loss";
            wrkflwi_output_loss->arguments.output.index = 1;
            wrkflwi_output_loss->output_format[0] = output_loss_format;

        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_softmax_loss, 0 }, { wrkflwi_backward_input, 0 } };
            di->workflow_item_create_function(&wrkflwi_softmax_loss_backprop, 2, inputs_descriptor, 1);
            wrkflwi_softmax_loss_backprop->type = NN_WORK_ITEM_TYPE_SOFTMAX_LOSS_BACKPROP;
            wrkflwi_softmax_loss_backprop->name = "softmax_loss_backprop";
            wrkflwi_softmax_loss_backprop->forward_item = wrkflwi_softmax_loss;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_softmax_loss_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_8_fc_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_8_fc_backprop->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP;
            wrkflwi_stage_8_fc_backprop->name = "fc8_backprop";
            wrkflwi_stage_8_fc_backprop->forward_item = wrkflwi_stage_8_fc;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_8_fc_backprop, 1 }, { wrkflwi_stage_8_fc_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_8_fc_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_8_fc_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_8_fc_update_backprop->name = "fc8_delta_update";
            wrkflwi_stage_8_fc_update_backprop->forward_item = wrkflwi_stage_8_fc;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_8_fc_backprop, 0 }, { wrkflwi_dropout_seed, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_7_fc_dropout_backprop, 3, inputs_descriptor, 1);
            wrkflwi_stage_7_fc_dropout_backprop->type = NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP;
            wrkflwi_stage_7_fc_dropout_backprop->name = "dropout7_backprop";
            wrkflwi_stage_7_fc_dropout_backprop->forward_item = wrkflwi_stage_7_fc_dropout;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_7_fc_dropout_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_7_fc_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_7_fc_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_7_fc_relu_backprop->name = "fc7relu_backprop";
            wrkflwi_stage_7_fc_relu_backprop->forward_item = wrkflwi_stage_7_fc_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_7_fc_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_7_fc_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_7_fc_backprop->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP;
            wrkflwi_stage_7_fc_backprop->name = "fc7_backprop";
            wrkflwi_stage_7_fc_backprop->forward_item = wrkflwi_stage_7_fc;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_7_fc_backprop, 1 }, { wrkflwi_stage_7_fc_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_7_fc_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_7_fc_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_7_fc_update_backprop->name = "fc7_delta_update";
            wrkflwi_stage_7_fc_update_backprop->forward_item = wrkflwi_stage_7_fc;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_7_fc_backprop, 0 }, { wrkflwi_dropout_seed, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_6_fc_dropout_backprop, 3, inputs_descriptor, 1);
            wrkflwi_stage_6_fc_dropout_backprop->type = NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP;
            wrkflwi_stage_6_fc_dropout_backprop->name = "dropout6_backprop";
            wrkflwi_stage_6_fc_dropout_backprop->forward_item = wrkflwi_stage_6_fc_dropout;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_6_fc_dropout_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_6_fc_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_6_fc_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_6_fc_relu_backprop->name = "fc6relu_backprop";
            wrkflwi_stage_6_fc_relu_backprop->forward_item = wrkflwi_stage_6_fc_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_6_fc_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_6_fc_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_6_fc_backprop->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP;
            wrkflwi_stage_6_fc_backprop->name = "fc6_backprop";
            wrkflwi_stage_6_fc_backprop->forward_item = wrkflwi_stage_6_fc;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_6_fc_backprop, 1 }, { wrkflwi_stage_6_fc_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_6_fc_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_6_fc_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_6_fc_update_backprop->name = "fc6_delta_update";
            wrkflwi_stage_6_fc_update_backprop->forward_item = wrkflwi_stage_6_fc;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = {{ wrkflwi_stage_6_fc_backprop, 0 }, {wrkflwi_stage_5_pool, 1}};
            di->workflow_item_create_function(&wrkflwi_stage_5_pool_backprop, 2, inputs_descriptor, 1);
            wrkflwi_stage_5_pool_backprop->type = NN_WORK_ITEM_TYPE_POOLING_BACKPROP;
            wrkflwi_stage_5_pool_backprop->name = "pool5_backprop";
            wrkflwi_stage_5_pool_backprop->forward_item = wrkflwi_stage_5_pool;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_pool_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_conv_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_5_conv_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_5_conv_relu_backprop->name = "c5relu_backprop";
            wrkflwi_stage_5_conv_relu_backprop->forward_item = wrkflwi_stage_5_conv_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_g1_subv_backprop, 1, &inputs_descriptor, 1); // view g1

            wrkflwi_stage_5_g1_subv_backprop->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_5_g1_subv_backprop->arguments.view.origin[0] = 0;
            wrkflwi_stage_5_g1_subv_backprop->arguments.view.origin[1] = 0;
            wrkflwi_stage_5_g1_subv_backprop->arguments.view.origin[2] = 0;
            wrkflwi_stage_5_g1_subv_backprop->name = "subv5g1_backprop";
            wrkflwi_stage_5_g1_subv_backprop->output_format[0] = nn::output_format{ 13, 13, 256/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_g2_subv_backprop, 1, &inputs_descriptor, 1); // view g2

            wrkflwi_stage_5_g2_subv_backprop->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_5_g2_subv_backprop->arguments.view.origin[0] = 0;
            wrkflwi_stage_5_g2_subv_backprop->arguments.view.origin[1] = 0;
            wrkflwi_stage_5_g2_subv_backprop->arguments.view.origin[2] = 256/2;
            wrkflwi_stage_5_g2_subv_backprop->name = "subv5g2_backprop";
            wrkflwi_stage_5_g2_subv_backprop->output_format[0] = nn::output_format{ 13, 13, 256/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_g1_subv_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_g1_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_5_g1_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_5_g1_conv_backprop->name = "c5g1_backprop";
            wrkflwi_stage_5_g1_conv_backprop->forward_item = wrkflwi_stage_5_g1_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_5_g1_conv_backprop, 1 }, { wrkflwi_stage_5_g1_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_5_g1_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_5_g1_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_5_g1_conv_update_backprop->name = "c5g1_delta_update";
            wrkflwi_stage_5_g1_conv_update_backprop->forward_item = wrkflwi_stage_5_g1_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_g2_subv_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_5_g2_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_5_g2_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_5_g2_conv_backprop->name = "c5g2_backprop";
            wrkflwi_stage_5_g2_conv_backprop->forward_item = wrkflwi_stage_5_g2_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_5_g2_conv_backprop, 1 }, { wrkflwi_stage_5_g2_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_5_g2_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_5_g2_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_5_g2_conv_update_backprop->name = "c5g2_delta_update";
            wrkflwi_stage_5_g2_conv_update_backprop->forward_item = wrkflwi_stage_5_g2_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_g1_conv_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g1_conv_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_4_g1_conv_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_4_g1_conv_relu_backprop->name = "c4g1relu_backprop";
            wrkflwi_stage_4_g1_conv_relu_backprop->forward_item = wrkflwi_stage_4_g1_conv_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_5_g2_conv_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g2_conv_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_4_g2_conv_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_4_g2_conv_relu_backprop->name = "c4g2relu_backprop";
            wrkflwi_stage_4_g2_conv_relu_backprop->forward_item = wrkflwi_stage_4_g2_conv_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_g1_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g1_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_4_g1_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_4_g1_conv_backprop->name = "c4g1_backprop";
            wrkflwi_stage_4_g1_conv_backprop->forward_item = wrkflwi_stage_4_g1_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_4_g1_conv_backprop, 1 }, { wrkflwi_stage_4_g1_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_4_g1_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_4_g1_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_4_g1_conv_update_backprop->name = "c4g1_delta_update";
            wrkflwi_stage_4_g1_conv_update_backprop->forward_item = wrkflwi_stage_4_g1_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_g2_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_g2_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_4_g2_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_4_g2_conv_backprop->name = "c4g2_backprop";
            wrkflwi_stage_4_g2_conv_backprop->forward_item = wrkflwi_stage_4_g2_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_4_g2_conv_backprop, 1 }, { wrkflwi_stage_4_g2_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_4_g2_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_4_g2_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_4_g2_conv_update_backprop->name = "c4g2_delta_update";
            wrkflwi_stage_4_g2_conv_update_backprop->forward_item = wrkflwi_stage_4_g2_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_4_g1_conv_backprop, 0 }, { wrkflwi_stage_4_g2_conv_backprop, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_4_merge_backprop, 2, inputs_descriptor, 1);

            wrkflwi_stage_4_merge_backprop->type = NN_WORK_ITEM_TYPE_MERGE;
            wrkflwi_stage_4_merge_backprop->arguments.forward_merge.axis = 2; // value 2 for z-axis
            wrkflwi_stage_4_merge_backprop->name = "merge4_backprop";
            wrkflwi_stage_4_merge_backprop->output_format[0] = nn::output_format{ 13, 13, 384 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_merge_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_3_conv_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_3_conv_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_3_conv_relu_backprop->name = "c3relu_backprop";
            wrkflwi_stage_3_conv_relu_backprop->forward_item = wrkflwi_stage_3_conv_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_3_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_3_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_3_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_3_conv_backprop->name = "c3_backprop";
            wrkflwi_stage_3_conv_backprop->forward_item = wrkflwi_stage_3_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_3_conv_backprop, 1 }, { wrkflwi_stage_3_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_3_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_3_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_3_conv_update_backprop->name = "c3_delta_update";
            wrkflwi_stage_3_conv_update_backprop->forward_item = wrkflwi_stage_3_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_3_conv_backprop, 0 }, { wrkflwi_stage_2_norm, 1 } };
            di->workflow_item_create_function(&wrkflwi_stage_2_norm_backprop, 2, inputs_descriptor, 1);
            wrkflwi_stage_2_norm_backprop->type = NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP;
            wrkflwi_stage_2_norm_backprop->name = "norm2_backprop";
            wrkflwi_stage_2_norm_backprop->forward_item = wrkflwi_stage_2_norm;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = {{ wrkflwi_stage_2_norm_backprop, 0 }, {wrkflwi_stage_2_pool, 1}};
            di->workflow_item_create_function(&wrkflwi_stage_2_pool_backprop, 2, inputs_descriptor, 1);
            wrkflwi_stage_2_pool_backprop->type = NN_WORK_ITEM_TYPE_POOLING_BACKPROP;
            wrkflwi_stage_2_pool_backprop->name = "pool2_backprop";
            wrkflwi_stage_2_pool_backprop->forward_item = wrkflwi_stage_2_pool;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_pool_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_conv_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_2_conv_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_2_conv_relu_backprop->name = "c2relu_backprop";
            wrkflwi_stage_2_conv_relu_backprop->forward_item = wrkflwi_stage_2_conv_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_g1_subv_backprop, 1, &inputs_descriptor, 1); // view g1

            wrkflwi_stage_2_g1_subv_backprop->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_2_g1_subv_backprop->arguments.view.origin[0] = 0;
            wrkflwi_stage_2_g1_subv_backprop->arguments.view.origin[1] = 0;
            wrkflwi_stage_2_g1_subv_backprop->arguments.view.origin[2] = 0;
            wrkflwi_stage_2_g1_subv_backprop->name = "subv2g1_backprop";
            wrkflwi_stage_2_g1_subv_backprop->output_format[0] = nn::output_format{ 27, 27, 256/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_g2_subv_backprop, 1, &inputs_descriptor, 1); // view g2

            wrkflwi_stage_2_g2_subv_backprop->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_2_g2_subv_backprop->arguments.view.origin[0] = 0;
            wrkflwi_stage_2_g2_subv_backprop->arguments.view.origin[1] = 0;
            wrkflwi_stage_2_g2_subv_backprop->arguments.view.origin[2] = 256/2;
            wrkflwi_stage_2_g2_subv_backprop->name = "subv2g2_backprop";
            wrkflwi_stage_2_g2_subv_backprop->output_format[0] = nn::output_format{ 27, 27, 256/2 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_g1_subv_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_g1_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_2_g1_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_2_g1_conv_backprop->name = "c2g1_backprop";
            wrkflwi_stage_2_g1_conv_backprop->forward_item = wrkflwi_stage_2_g1_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_2_g1_conv_backprop, 1 }, { wrkflwi_stage_2_g1_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_2_g1_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_2_g1_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_2_g1_conv_update_backprop->name = "c2g1_delta_update";
            wrkflwi_stage_2_g1_conv_update_backprop->forward_item = wrkflwi_stage_2_g1_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_g2_subv_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_g2_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_2_g2_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_2_g2_conv_backprop->name = "c2g2_backprop";
            wrkflwi_stage_2_g2_conv_backprop->forward_item = wrkflwi_stage_2_g2_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_2_g2_conv_backprop, 1 }, { wrkflwi_stage_2_g2_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_2_g2_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_2_g2_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_2_g2_conv_update_backprop->name = "c2g2_delta_update";
            wrkflwi_stage_2_g2_conv_update_backprop->forward_item = wrkflwi_stage_2_g2_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_2_g1_conv_backprop, 0 }, { wrkflwi_stage_2_g2_conv_backprop, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_2_merge_backprop, 2, inputs_descriptor, 1);
            wrkflwi_stage_2_merge_backprop->type = NN_WORK_ITEM_TYPE_MERGE;
            wrkflwi_stage_2_merge_backprop->arguments.forward_merge.axis = 2; // value 2 for z-axis
            wrkflwi_stage_2_merge_backprop->name = "merge2_backprop";
            wrkflwi_stage_2_merge_backprop->output_format[0] = nn::output_format{ 27, 27, 96 };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_2_merge_backprop, 0 }, { wrkflwi_stage_1_norm, 1 } };
            di->workflow_item_create_function(&wrkflwi_stage_1_norm_backprop, 2, inputs_descriptor, 1);
            wrkflwi_stage_1_norm_backprop->type = NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP;
            wrkflwi_stage_1_norm_backprop->name = "norm1_backprop";
            wrkflwi_stage_1_norm_backprop->forward_item = wrkflwi_stage_1_norm;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = {{ wrkflwi_stage_1_norm_backprop, 0 }, {wrkflwi_stage_1_pool, 1}};
            di->workflow_item_create_function(&wrkflwi_stage_1_pool_backprop, 2, inputs_descriptor, 1);
            wrkflwi_stage_1_pool_backprop->type = NN_WORK_ITEM_TYPE_POOLING_BACKPROP;
            wrkflwi_stage_1_pool_backprop->name = "pool1_backprop";
            wrkflwi_stage_1_pool_backprop->forward_item = wrkflwi_stage_1_pool;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_pool_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_conv_relu_backprop, 1, &inputs_descriptor, 1);
            wrkflwi_stage_1_conv_relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;
            wrkflwi_stage_1_conv_relu_backprop->name = "c1relu_backprop";
            wrkflwi_stage_1_conv_relu_backprop->forward_item = wrkflwi_stage_1_conv_relu;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_conv_relu_backprop, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_conv_backprop, 1, &inputs_descriptor, 3);
            wrkflwi_stage_1_conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;
            wrkflwi_stage_1_conv_backprop->name = "c1_backprop";
            wrkflwi_stage_1_conv_backprop->forward_item = wrkflwi_stage_1_conv;
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { wrkflwi_stage_1_conv_backprop, 1 }, { wrkflwi_stage_1_conv_backprop, 2 }, { wrkflwi_learning_rate, 0 }, { wrkflwi_execution_mode, 0 } };
            di->workflow_item_create_function(&wrkflwi_stage_1_conv_update_backprop, 4, inputs_descriptor, 0);
            wrkflwi_stage_1_conv_update_backprop->type = NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS;

            wrkflwi_stage_1_conv_update_backprop->name = "c1_delta_update";
            wrkflwi_stage_1_conv_update_backprop->forward_item = wrkflwi_stage_1_conv;
        }

        // -------------------------------------------------------------------------------------------
        // END of backward workflow stages definition
        // -------------------------------------------------------------------------------------------
        workflow->input[1] = wrkflwi_backward_input;
        workflow->input[4] = wrkflwi_learning_rate;
        workflow->output[1] = wrkflwi_output_loss;

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
        workflow_builder_caffenet_float_training builder;
        attach() {
            workflow_builder::instance().add("caffenet_float_training", &builder);
        }
    };

    attach attach_;
}
