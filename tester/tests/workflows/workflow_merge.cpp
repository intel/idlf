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
    input0,
    input1,
    merge,
    output,
    last_workflow_item = output
};

class workflow_merge : public workflows_for_tests_base {
public:
    workflow_merge( std::vector< uint32_t > &input_sizes,
                    std::vector< uint32_t > &output_sizes ) {
        for(auto wi : workflow_layer) wi = nullptr;

        if( (input_sizes.size()  == 3) &&
            (output_sizes.size() == 3)) {
            this->input_sizes  = input_sizes;
            this->output_sizes = output_sizes;
        } else
            throw std::runtime_error("Error: Invalid number of dimnesions for workflow");
}

    bool is_valid() { return error_.empty(); }

private:
    std::string error_;
    std::vector< uint32_t > input_sizes;
    std::vector< uint32_t > output_sizes;

    // pointers to successive workflow parts
    nn_workflow_item_t        *workflow_layer[last_workflow_item + 1];

    nn_workflow_t             *workflow = nullptr;
    nn_device_interface_0_t   *di = nullptr;

public:

    virtual nn_workflow_t *init_test_workflow( nn_device_interface_0_t *_di ) {

        if(!is_valid()) throw std::runtime_error( error_ );

        for(auto wi : workflow_layer) wi = nullptr;

        this->di = _di;

        di->workflow_create_function( &workflow, 2, 1 );

        // STAGE 0
        { // input 1
            di->workflow_item_create_function( &workflow_layer[input0], 0, nullptr, 1 );

            workflow_layer[input0]->type = NN_WORK_ITEM_TYPE_INPUT;
            workflow_layer[input0]->arguments.input.index = 0;
            workflow_layer[input0]->output_format[0] = nn::output_format { input_sizes.at(0), input_sizes.at(1), input_sizes.at(2) };

        }
        { // input 2
            di->workflow_item_create_function( &workflow_layer[input1], 0, nullptr, 1 );

            workflow_layer[input1]->type = NN_WORK_ITEM_TYPE_INPUT;
            workflow_layer[input1]->arguments.input.index = 1;
            workflow_layer[input1]->output_format[0] = nn::output_format { input_sizes.at(0), input_sizes.at(1), input_sizes.at(2) };
        }

        // STAGE 1 merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = {{ workflow_layer[input0], 0 }, { workflow_layer[input1], 0 }};
            di->workflow_item_create_function( &workflow_layer[merge], 2, inputs_descriptor, 1 );

            workflow_layer[merge]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_layer[merge]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_layer[merge]->output_format[0] = nn::output_format { output_sizes.at(0), output_sizes.at(1), output_sizes.at(2) };
        }
        // ------------------------------------------------------------------------------------------
        // STAGE 2 output
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[merge], 0 };
            di->workflow_item_create_function( &workflow_layer[output], 1, &inputs_descriptor, 1 );

            workflow_layer[output]->type = NN_WORK_ITEM_TYPE_OUTPUT;

            workflow_layer[output]->output_format[0] = nn::output_format { output_sizes.at(0), output_sizes.at(1), output_sizes.at(2) };
        }
    // -------------------------------------------------------------------------------------------
    // END of workflow stages definition
    workflow->input[0]  = workflow_layer[input0];
    workflow->input[1]  = workflow_layer[input1];
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
    const uint32_t x = 24, y = 24, z = 16;
    std::vector<uint32_t> in_size  {x, y, z/2};
    std::vector<uint32_t> out_size {x, y, z};

    struct attach {
        workflow_merge *test_workflow;
        attach( std::vector<uint32_t> in_size, std::vector<uint32_t> out_size, std::string name )
            : test_workflow( new workflow_merge( in_size, out_size ))
        {
            workflows_for_tests::instance().add( name, test_workflow );
        }
        ~attach() { delete test_workflow; }
    };

    attach attach_( in_size, out_size, "workflow_merge" );
}
