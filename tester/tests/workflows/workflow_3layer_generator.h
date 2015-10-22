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

#pragma once
#include "tester/common/workflows_for_tests.h"

template < typename T >    // T implements middle layer
class workflow_3layer_generator : public workflows_for_tests_base {
public:
    workflow_3layer_generator( std::vector< uint32_t > &input_sizes,
                               std::vector< uint32_t > &output_sizes) {

        for(auto wi : workflow_layer) wi = nullptr;
        if( input_sizes.size()        && 
            output_sizes.size()       &&
            (input_sizes.size()  < 4) &&
            (output_sizes.size() < 4)) {
        this->input_sizes  = input_sizes;
        this->output_sizes = output_sizes;
        } else
            throw std::runtime_error("Error: Invalid number of dimnesions for workflow");
    }

    bool is_valid() { return error_.empty(); }

private:
    enum workflow_layers {
        input,
        mid,
        output,
        last_workflow_item = output
    };
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

        di->workflow_create_function( &workflow, 1, 1 );

        // STAGE 0 (input)
        {
            di->workflow_item_create_function( &workflow_layer[input], 0, nullptr, 1 );

            workflow_layer[input]->type = NN_WORK_ITEM_TYPE_INPUT;
            workflow_layer[input]->arguments.input.index = 0;

            if(1 == input_sizes.size()) {
               workflow_layer[input]->output_format[0] = nn::output_format { input_sizes.at( 0 ) };
            } else if (2 == input_sizes.size()) {
                workflow_layer[input]->output_format[0] = nn::output_format { input_sizes.at(0), input_sizes.at(1) };
            } else if (3 == input_sizes.size()) {
                workflow_layer[input]->output_format[0] = nn::output_format { input_sizes.at(0), input_sizes.at(1), input_sizes.at(2) };
            } else
                throw std::runtime_error("Workflow initialization failed, wrong number of dimnesions.");
        }

        // STAGE middle
        {
            workflow_layer[mid] = T::create_layer( workflow_layer[input], di);
        }
        // ------------------------------------------------------------------------------------------
        // STAGE output
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_layer[mid], 0 };
            di->workflow_item_create_function( &workflow_layer[output], 1, &inputs_descriptor, 1 );

            workflow_layer[output]->type = NN_WORK_ITEM_TYPE_OUTPUT;

            if(1 == input_sizes.size()) {
                workflow_layer[output]->output_format[0] = nn::output_format { output_sizes.at(0) };
            } else if (2 == input_sizes.size()) {
                workflow_layer[output]->output_format[0] = nn::output_format { output_sizes.at(0), output_sizes.at(1) };
            } else if (3 == input_sizes.size()) {
                workflow_layer[output]->output_format[0] = nn::output_format { output_sizes.at(0), output_sizes.at(1), output_sizes.at(2) };
            } else
                throw std::runtime_error("Workflow initialization failed, wrong number of dimnesions.");
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
