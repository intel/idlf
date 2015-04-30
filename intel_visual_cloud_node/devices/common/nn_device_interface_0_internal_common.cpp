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

#include "../api/nn_device_interface_0.h"
#include <cstring>
#include <cassert>

/* create empty workflow */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_create_0_function(
    nn_workflow_t *    *workflow,       /* workflow to be created */
    uint32_t            input_count,    /* number of inputs in created workflow */
    uint32_t            output_count    /* number of outputs in created workflow */
) {
    // only 1 input & 1 output are currently supported
    if(!workflow)       return NN_API_STATUS_ERROR_INVALID_POINTER;
    if( input_count==0 ||  input_count>16) return NN_API_STATUS_ERROR_INVALID_INPUT_COUNT;
    if(output_count==0 || output_count>16) return NN_API_STATUS_ERROR_INVALID_OUTPUT_COUNT;
    try {
        const size_t  input_size = sizeof(nn_workflow_item_t *)* input_count;
        const size_t output_size = sizeof(nn_workflow_item_t *)*output_count;
        const size_t buffer_size = sizeof(nn_workflow_t)+input_size+output_size;
        uint8_t *buffer = new uint8_t[buffer_size]();
        *workflow = reinterpret_cast<nn_workflow_t *>(buffer);
        buffer += sizeof(nn_workflow_t);
        *const_cast<nn_workflow_item_t *const **>(&(*workflow)-> input) = reinterpret_cast<nn_workflow_item_t **>(buffer);
        buffer += input_size;
        *const_cast<nn_workflow_item_t *const **>(&(*workflow)->output) = reinterpret_cast<nn_workflow_item_t **>(buffer);
        *const_cast<uint32_t *>(&(*workflow)->input_count)  =  input_count;
        *const_cast<uint32_t *>(&(*workflow)->output_count) = output_count;
        return NN_API_STATUS_OK;
    }
    catch(...) {
        return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
    }
}

/* delete workflow */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_delete_0_function(
    nn_workflow_t      *workflow        /* workflow to delete */
) {
    if(workflow) {
        delete[] reinterpret_cast<uint8_t *>(workflow);
        return NN_API_STATUS_OK;
    } else {
        return NN_API_STATUS_ERROR_INVALID_POINTER;
    }
}

/* create empty work item */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_create_0_function(
    nn_workflow_item_t**item,           /* resulting workflow item */
    uint32_t            input_count,    /* count of inputs */
    nn_workflow_item_t**input           /* pointer to inputs */
) {
    try {
        nn_workflow_item_t *workflow_item = reinterpret_cast<nn_workflow_item_t *>(new uint8_t[sizeof(nn_workflow_item_t)]());

        workflow_item->name = "unnamed";

        workflow_item->input_count = input_count;
        if(input_count) {
            workflow_item->input = new nn_workflow_item_t*[input_count];
            std::memcpy(workflow_item->input, input, input_count*sizeof(nn_workflow_item_t *));
        } else {
            workflow_item->input = nullptr;
        }
        // add current item to use list of all it's input nodes
        for(auto index=0u; index<input_count; ++index) {
            // lambda for finding out if entry already exists in use list of an item
            auto already_exists = [](nn_workflow_item_t *item, nn_workflow_item_t *entry) -> bool {
                for(auto index=0u; index<item->use_count; ++index)
                    if(item->use[index]==entry) return true;
                return false;
            }; // lambda

            nn_workflow_item_t *input_item = workflow_item->input[index];
            if(!already_exists(input_item, workflow_item)) {
                // current node is not on the use list of it's input; re-allocate input's buffer and add it
                nn_workflow_item_t **use = new nn_workflow_item_t *[input_item->use_count+1];
                std::memcpy(use, input_item->use, input_item->use_count*sizeof(nn_workflow_item_t *));
                use[input_item->use_count] = workflow_item;
                delete[] input_item->use;
                *const_cast<nn_workflow_item_t ***>(&input_item->use)       = use;
                *const_cast<uint32_t *>            (&input_item->use_count) = input_item->use_count+1;
            }
        }
        *item = workflow_item;
        return NN_API_STATUS_OK;
    }
    catch(...) {
        return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
    }
}

/* delete workflow item */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_delete_0_function(
    nn_workflow_item_t *item            /* work item to be deleted */
) {
    if(item) {
        try {
            // remove 'item' from use lists at all of its input items
            for(auto input_index=0u; input_index<item->input_count; ++input_index) {
                auto input_item = item->input[input_index];
                assert(input_item!=nullptr);
                // shrink use list of specific input
                const bool non_zero_uses = input_item->use_count>1;
                const uint32_t use_count = non_zero_uses ? input_item->use_count-1 : 0;
                nn_workflow_item **use = non_zero_uses ? new nn_workflow_item *[use_count] : nullptr;
                if(non_zero_uses) {
                    uint32_t at=0u;
                    for(auto use_index=0u; use_index<input_item->use_count; ++use_index) {
                        if(input_item->use[use_index]==item) continue;
                        use[at++] = input_item->use[use_index];
                    }
                }
                delete[] input_item->use;
                *const_cast<nn_workflow_item_t ***>(&input_item->use)       = use;
                *const_cast<uint32_t *>            (&input_item->use_count) = use_count;
            }
            // remove 'item' from input lists at all of its use items
            for(auto use_index=0u; use_index<item->use_count; ++use_index) {
                auto use_item = item->use[use_index];
                assert(use_item!=nullptr);
                // shrink input list of specific use
                const bool non_zero_inputs = use_item->input_count>1;
                const uint32_t input_count = non_zero_inputs ? use_item->input_count-1 : 0;
                nn_workflow_item **input = non_zero_inputs ? new nn_workflow_item *[use_item->input_count-1] : nullptr;
                if(non_zero_inputs) {
                    uint32_t at=0u;
                    for(auto input_index=0u; input_index<use_item->input_count; ++input_index) {
                        if(use_item->input[input_index]) continue;
                        input[at++] = use_item->input[input_index];
                    }
                }
                delete[] use_item->input;
                *const_cast<nn_workflow_item_t ***>(&use_item->input)       = input;
                *const_cast<uint32_t *>            (&use_item->input_count) = input_count;
            }
            delete[] item->input;
            delete[] item->use;
            delete[] reinterpret_cast<uint8_t *>(item);
        }
        catch(...) {
            return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
        }
    }
    return NN_API_STATUS_OK;
}

NN_API_STATUS NN_API_CALL_CONVENTION nn_translate_api_status_0_function(
    NN_API_STATUS       status,          /* status code to translate */
    char*              *brief,           /* one-line explanation */
    char*              *detailed         /* multi-line explanation */
) {
    return NN_API_STATUS_ERROR_OTHER;
}
