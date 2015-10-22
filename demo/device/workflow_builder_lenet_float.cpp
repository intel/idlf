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

static NN_WORKLOAD_DATA_TYPE in_formats[] =
        { NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH };

static NN_WORKLOAD_DATA_TYPE out_formats[] =
        { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH };


class workflow_builder_lenet_float: public workflow_builder_base
{

public:

    workflow_builder_lenet_float() : workflow_builder_base(28) {
        //TODO: Check if weights files does exist in given directory?
    }

    bool is_valid() { return error_.empty(); }

    virtual NN_WORKLOAD_DATA_TYPE* get_input_formats() {return in_formats;}
    virtual NN_WORKLOAD_DATA_TYPE* get_output_formats() {return out_formats;}

private:
    std::string error_;

    // pointers to successive workflow parts
    nn_workflow_item_t
        *wrkflwi_input,
        *wrkflwi_stage_1_conv,
        *wrkflwi_stage_1_pool,
        *wrkflwi_stage_1_subv,
        *wrkflwi_stage_2_conv,
        *wrkflwi_stage_2_pool,
        *wrkflwi_stage_3_fc,
        *wrkflwi_stage_4_fc,
        *wrkflwi_softmax,
        *wrkflwi_output;

    // pointers to <nn_workload_data>s containing weights and biases;
    nn::data<float>
        *nnwrkld_conv1_weights,
        *nnwrkld_conv1_biases,
        *nnwrkld_conv2_weights,
        *nnwrkld_conv2_biases,
        *nnwrkld_fc1_weights,
        *nnwrkld_fc1_biases,
        *nnwrkld_fc2_weights,
        *nnwrkld_fc2_biases;

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
        di->workflow_item_delete_function(wrkflwi_stage_1_conv);
        di->workflow_item_delete_function(wrkflwi_stage_1_pool);
        di->workflow_item_delete_function(wrkflwi_stage_1_subv);
        di->workflow_item_delete_function(wrkflwi_stage_2_conv);
        di->workflow_item_delete_function(wrkflwi_stage_2_pool);
        di->workflow_item_delete_function(wrkflwi_stage_3_fc);
        di->workflow_item_delete_function(wrkflwi_stage_4_fc);
        di->workflow_item_delete_function(wrkflwi_softmax);
        di->workflow_item_delete_function(wrkflwi_output);

        di->workflow_delete_function(workflow);

        delete nnwrkld_conv1_weights;
        delete nnwrkld_conv1_biases;
        delete nnwrkld_conv2_weights;
        delete nnwrkld_conv2_biases;
        delete nnwrkld_fc1_weights;
        delete nnwrkld_fc1_biases;
        delete nnwrkld_fc2_weights;
        delete nnwrkld_fc2_biases;
        delete di;
    }

    virtual nn_workflow_t *init_workflow(nn_device_interface_0_t *di){

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
            nnwrkld_conv1_weights = load_biases_or_weights("weights_lenet/conv1.nn");
            nnwrkld_conv1_biases = load_biases_or_weights("weights_lenet/conv1_bias.nn");
            nnwrkld_conv2_weights = load_biases_or_weights("weights_lenet/conv2.nn");
            nnwrkld_conv2_biases = load_biases_or_weights("weights_lenet/conv2_bias.nn");
            nnwrkld_fc1_weights = load_biases_or_weights("weights_lenet/ip1.nn");
            nnwrkld_fc1_biases = load_biases_or_weights("weights_lenet/ip1_bias.nn");
            nnwrkld_fc2_weights = load_biases_or_weights("weights_lenet/ip2.nn");
            nnwrkld_fc2_biases = load_biases_or_weights("weights_lenet/ip2_bias.nn");
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
        //         output: 28x28x3
        {
            di->workflow_item_create_function(&wrkflwi_input, 0, nullptr, 1);

            wrkflwi_input->type = NN_WORK_ITEM_TYPE_INPUT;
            wrkflwi_input->arguments.input.index = 0;
            wrkflwi_input->output_format[0].format = NN_DATA_FORMAT_2D;
            wrkflwi_input->output_format[0].format_3d ={ { img_size, img_size} };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 01
        //           convo: 5x5 stride 1x1; no-activation; output: 24x24x20
        //         maxpool: 2x2 stride 2x2;
        //          output: 12x12x20
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_input, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_conv, 1, &inputs_descriptor, 1);

            wrkflwi_stage_1_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_1_conv->name = "c1";

            wrkflwi_stage_1_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            wrkflwi_stage_1_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            // We have weights, biases for 20 filters , but we want to have for four more filters so lets add padding
            wrkflwi_stage_1_conv->arguments.forward_convolution.weights = nn_data_extend_weights_by_padding(nnwrkld_conv1_weights,1,24);
            wrkflwi_stage_1_conv->arguments.forward_convolution.biases = nn_data_extend_biases_by_padding(nnwrkld_conv1_biases,24);

            wrkflwi_stage_1_conv->arguments.forward_convolution.center_offset[0] = 0;
            wrkflwi_stage_1_conv->arguments.forward_convolution.center_offset[1] = 0;

            wrkflwi_stage_1_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_1_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_1_conv->output_format[0].format = NN_DATA_FORMAT_3D;
            // It should be 20 output FM , but we do support only case when output FM number is divisble by 8
            wrkflwi_stage_1_conv->output_format[0].format_3d ={ { 24, 24, 24 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_conv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_pool, 1, &inputs_descriptor, 1);

            wrkflwi_stage_1_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_1_pool->name = "p1";

            wrkflwi_stage_1_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            wrkflwi_stage_1_pool->arguments.forward_pooling.size[0] = 2;
            wrkflwi_stage_1_pool->arguments.forward_pooling.size[1] = 2;
            wrkflwi_stage_1_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_1_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_1_pool->output_format[0].format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_1_pool->output_format[0].format_3d ={ { 12, 12, 24 } };
        }
        // view
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_pool, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_1_subv, 1, &inputs_descriptor, 1); // view

            wrkflwi_stage_1_subv->type = NN_WORK_ITEM_TYPE_VIEW;
            wrkflwi_stage_1_subv->arguments.view.origin[0] = 0;
            wrkflwi_stage_1_subv->arguments.view.origin[1] = 0;
            wrkflwi_stage_1_subv->arguments.view.origin[2] = 0;

            wrkflwi_stage_1_subv->output_format[0].format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_1_subv->output_format[0].format_3d ={ { 12, 12, 20 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 02
        //           convo: 5x5 stride 1x1; no-activation; output: 8x8x50
        //         maxpool: 2x2 stride 2x2;
        //          output: 4x4x50

        // convolution 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_1_subv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_conv, 1, &inputs_descriptor, 1);

            wrkflwi_stage_2_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            wrkflwi_stage_2_conv->name = "c2";

            wrkflwi_stage_2_conv->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            wrkflwi_stage_2_conv->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            wrkflwi_stage_2_conv->arguments.forward_convolution.weights = nn_data_extend_weights_by_padding(nnwrkld_conv2_weights,20,56);
            wrkflwi_stage_2_conv->arguments.forward_convolution.biases = nn_data_extend_biases_by_padding(nnwrkld_conv2_biases,56);

            wrkflwi_stage_2_conv->arguments.forward_convolution.center_offset[0] = 0;
            wrkflwi_stage_2_conv->arguments.forward_convolution.center_offset[1] = 0;

            wrkflwi_stage_2_conv->arguments.forward_convolution.stride[0] = 1;
            wrkflwi_stage_2_conv->arguments.forward_convolution.stride[1] = 1;

            wrkflwi_stage_2_conv->output_format[0].format = NN_DATA_FORMAT_3D;
            // It should be 50 output FM , but we do support only case when output FM number is divisble by 8
            wrkflwi_stage_2_conv->output_format[0].format_3d ={ { 8, 8, 56 } };
        }

        // maxpool: 2x2 stride 2x2;
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_conv, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_2_pool, 1, &inputs_descriptor, 1); // pooling

            wrkflwi_stage_2_pool->type = NN_WORK_ITEM_TYPE_POOLING;
            wrkflwi_stage_2_pool->name = "p2";

            wrkflwi_stage_2_pool->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

            wrkflwi_stage_2_pool->arguments.forward_pooling.size[0] = 2;
            wrkflwi_stage_2_pool->arguments.forward_pooling.size[1] = 2;

            wrkflwi_stage_2_pool->arguments.forward_pooling.stride[0] = 2;
            wrkflwi_stage_2_pool->arguments.forward_pooling.stride[1] = 2;

            wrkflwi_stage_2_pool->output_format[0].format = NN_DATA_FORMAT_3D;
            wrkflwi_stage_2_pool->output_format[0].format_3d ={ { 4, 4, 56 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 03
        //            full: ReLU
        //          output: 500
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_2_pool, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_3_fc, 1, &inputs_descriptor, 1);

            wrkflwi_stage_3_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_3_fc->name = "fc1";

            wrkflwi_stage_3_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_RELU;


            // Generated weights if taken from caffe , are in 2D format while we need them in 4d format
            nn::data<float>* nnwrkld_fc1_converted_weights = nn_data_convert_weights_2D_to_4D(nnwrkld_fc1_weights,
                                                                                              4,
                                                                                              4,
                                                                                              50,
                                                                                              static_cast<uint32_t>(nnwrkld_fc1_weights->size[1]));
            // release original weights
            delete nnwrkld_fc1_weights;
            // Extend weights' depth of FC layer to match extended weights input
            nnwrkld_fc1_weights = nn_data_extend_weights_by_padding(nnwrkld_fc1_converted_weights,
                                                                    56,
                                                                    static_cast<uint32_t>(nnwrkld_fc1_converted_weights->size[3]));
            delete nnwrkld_fc1_converted_weights;
            nnwrkld_fc1_converted_weights = nullptr;

            wrkflwi_stage_3_fc->arguments.forward_fully_connected.weights = nnwrkld_fc1_weights;
            wrkflwi_stage_3_fc->arguments.forward_fully_connected.biases = nnwrkld_fc1_biases;

            wrkflwi_stage_3_fc->output_format[0].format = NN_DATA_FORMAT_1D;
            wrkflwi_stage_3_fc->output_format[0].format_1d ={ { 500 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 04
        //            full: ;
        //          output: 10
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_3_fc, 0 };
            di->workflow_item_create_function(&wrkflwi_stage_4_fc, 1, &inputs_descriptor, 1);

            wrkflwi_stage_4_fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            wrkflwi_stage_4_fc->name = "fc2";

            wrkflwi_stage_4_fc->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            wrkflwi_stage_4_fc->arguments.forward_fully_connected.weights = nnwrkld_fc2_weights;
            wrkflwi_stage_4_fc->arguments.forward_fully_connected.biases = nnwrkld_fc2_biases;

            wrkflwi_stage_4_fc->output_format[0].format = NN_DATA_FORMAT_1D;
            wrkflwi_stage_4_fc->output_format[0].format_1d ={ { 10 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 05 (softmax)
        //          output: 10
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_stage_4_fc, 0 };
            di->workflow_item_create_function(&wrkflwi_softmax, 1, &inputs_descriptor, 1);

            wrkflwi_softmax->type = NN_WORK_ITEM_TYPE_SOFTMAX;

            wrkflwi_softmax->output_format[0].format = NN_DATA_FORMAT_1D;
            wrkflwi_softmax->output_format[0].format_1d ={ { 10 } };
        }

        // ------------------------------------------------------------------------------------------
        // STAGE 6 (output)
        //          output: 10
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { wrkflwi_softmax, 0 };
            di->workflow_item_create_function(&wrkflwi_output, 1, &inputs_descriptor, 1);

            wrkflwi_output->type = NN_WORK_ITEM_TYPE_OUTPUT;

            wrkflwi_output->output_format[0].format = NN_DATA_FORMAT_1D;
            wrkflwi_output->output_format[0].format_1d ={ { 10 } };

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
        workflow_builder_lenet_float builder;
        attach() {
            workflow_builder::instance().add("lenet_float", &builder);
        }
    };

    attach attach_;
}
