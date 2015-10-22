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

enum workflow_layers {
    input,
    relu,
    output,
    last_workflow_item = output
};

class workflow_for_testing_float_relu_float : public workflows_for_tests_base {
public:
    workflow_for_testing_float_relu_float() {
        for(auto wi : workflow_layer) wi = nullptr;
    }

    bool is_valid() { return error_.empty(); }

private:
    std::string error_;

    uint16_t relu_length = 1000;

    // pointers to successive workflow parts
    nn_workflow_item_t        *workflow_layer[last_workflow_item + 1];

    nn_workflow_t             *workflow = nullptr;
    nn_device_interface_0_t   *di = nullptr;

public:

    virtual nn_workflow_t *init_test_workflow( nn_device_interface_0_t *_di ) {

        if(!is_valid()) throw std::runtime_error( error_ );

        for(auto wi : workflow_layer) wi = nullptr;

        this->di = _di;

        di->workflow_create_function( &workflow, 1, 1 );

        // STAGE 0 (input)
        {
            di->workflow_item_create_function( &workflow_layer[input], 0, nullptr, 1 );

            workflow_layer[input]->type = NN_WORK_ITEM_TYPE_INPUT;
            workflow_layer[input]->arguments.input.index = 0;
            workflow_layer[input]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[input]->output_format[0].format_1d = { { relu_length } };
        }

        // STAGE 1 relu
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[input], 0 };
            di->workflow_item_create_function( &workflow_layer[relu], 1, &inputs_descriptor, 1 );

            workflow_layer[relu]->type = NN_WORK_ITEM_TYPE_RELU;

            workflow_layer[relu]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_layer[relu]->output_format[0].format_1d = { { relu_length } };
        }
        // ------------------------------------------------------------------------------------------
        // STAGE 2 output
    {
        nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[relu], 0 };
        di->workflow_item_create_function( &workflow_layer[output], 1, &inputs_descriptor, 1 );

        workflow_layer[output]->type = NN_WORK_ITEM_TYPE_OUTPUT;

        workflow_layer[output]->output_format[0].format = NN_DATA_FORMAT_1D;
        workflow_layer[output]->output_format[0].format_3d = { { relu_length } };

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
    }
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran befor main execution starts.
// The sole function of this construction is attaching this workflow builder to
// library of workflow builders (singleton command pattern).
namespace {
    struct attach {
        workflow_for_testing_float_relu_float test_workflow;
        attach() {
            workflows_for_tests::instance().add( "workflow_for_testing_float_relu", &test_workflow );
        }
    };

    attach attach_;
}
