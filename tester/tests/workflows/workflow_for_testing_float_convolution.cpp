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

#include "tester/common/test_common_tools.h"
#include "tester/common/workflows_for_tests.h"

enum workflow_layers {
    input,
    conv,
    output,
    last_workflow_item = output
};

enum workflow_layer_factor {
    conv_weights,
    conv_biases,
    last_weights_and_biases = conv_biases
};

class workflow_for_testing_float_convolution : public workflows_for_tests_base {
public:
    workflow_for_testing_float_convolution() {
        for(auto wi : workflow_layer) wi = nullptr;
        for(auto wlf : workflow_layer_factor) wlf = nullptr;
    }

    bool is_valid() { return error_.empty(); }

private:
    std::string error_;

    uint32_t img_size = 227; //TO DO template
    uint32_t num_features_map = 8; //TO DO template
    uint32_t z = 2; //TO DO template

    // pointers to successive workflow parts
    nn_workflow_item_t        *workflow_layer[last_workflow_item + 1];

    // pointers to nn_datas containing weights and biases;
    nn::data<float>           *workflow_layer_factor[last_weights_and_biases + 1];

    nn_workflow_t             *workflow = nullptr;
    nn_device_interface_0_t   *di = nullptr;

public:

    virtual nn_workflow_t *init_test_workflow( nn_device_interface_0_t *_di ) {

        if(!is_valid()) throw std::runtime_error( error_ );

        for(auto wi : workflow_layer) wi = nullptr;
        for(auto wlf : workflow_layer_factor) wlf = nullptr;

        this->di = _di;
        // load nn:data factors (weights and biases) for successive layers
        workflow_layer_factor[conv_weights] = new nn::data<float>( 3, 3, z, num_features_map );
        nn_data_populate( nn::data_cast<float, 0>(workflow_layer_factor[conv_weights]),
            0.0f,
            255.0f);

        workflow_layer_factor[conv_biases] = new nn::data<float>( num_features_map );
        nn_data_populate( nn::data_cast<float, 0>(workflow_layer_factor[conv_biases]),
            0.0f,
            255.0f );

        for(auto wlf : workflow_layer_factor)
            if(wlf == nullptr)
                std::runtime_error( "One or more of workflow factor was not loaded" );

        di->workflow_create_function( &workflow, 1, 1 );

        // STAGE 01 (input)
        {
            di->workflow_item_create_function( &workflow_layer[input], 0, nullptr, 1 );

            workflow_layer[input]->type = NN_WORK_ITEM_TYPE_INPUT;
            workflow_layer[input]->arguments.input.index = 0;
            workflow_layer[input]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[input]->output_format[0].format_3d = { { img_size, img_size, z } };
        }

        // STAGE 02
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[input], 0 };
            di->workflow_item_create_function( &workflow_layer[conv], 1, &inputs_descriptor, 1 );

            workflow_layer[conv]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_layer[conv]->name = "conv";
            workflow_layer[conv]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_layer[conv]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_layer[conv]->arguments.forward_convolution.weights = workflow_layer_factor[conv_weights];
            workflow_layer[conv]->arguments.forward_convolution.biases = workflow_layer_factor[conv_biases];

            workflow_layer[conv]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_layer[conv]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_layer[conv]->arguments.forward_convolution.stride[0] = 1;
            workflow_layer[conv]->arguments.forward_convolution.stride[1] = 1;

            workflow_layer[conv]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_layer[conv]->output_format[0].format_3d = { { img_size, img_size, num_features_map } };
        }
        // ------------------------------------------------------------------------------------------
        // STAGE 03 (output)
    {
        nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[conv], 0 };
        di->workflow_item_create_function( &workflow_layer[output], 1, &inputs_descriptor, 1 );

        workflow_layer[output]->type = NN_WORK_ITEM_TYPE_OUTPUT;

        workflow_layer[output]->output_format[0].format = NN_DATA_FORMAT_3D;
        workflow_layer[output]->output_format[0].format_3d = { { img_size, img_size, num_features_map } };

    }
    // -------------------------------------------------------------------------------------------
    // END of workflow stages definition
    workflow->input[0] = workflow_layer[input];
    workflow->output[0] = workflow_layer[output];
    // -------------------------------------------------------------------------------------------

    return workflow;
    }

    void cleanup() {
        if(!is_valid()) throw std::runtime_error( error_ );

        for(auto wl : workflow_layer)
            di->workflow_item_delete_function( wl );

        di->workflow_delete_function( workflow );

        for(auto wb : workflow_layer_factor)
            if(wb != nullptr) delete wb;

    }
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran befor main execution starts.
// The sole function of this construction is attaching this workflow builder to
// library of workflow builders (singleton command pattern).
namespace {
    struct attach {
        workflow_for_testing_float_convolution test_workflow;
        attach() {
            workflows_for_tests::instance().add( "workflow_for_testing_float_convolution", &test_workflow );
        }
    };

    attach attach_;
}
