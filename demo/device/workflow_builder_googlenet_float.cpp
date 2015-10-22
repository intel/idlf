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
{ NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH };

static NN_WORKLOAD_DATA_TYPE out_formats[] =
{ NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH };

enum  workflow_items {
    input,
    mean,
    conv1_7x7_s2,
    pool1_3x3_s2,
    pool1_norm1,
    conv2_3x3_reduce,
    conv2_3x3,
    conv2_norm2,
    pool2_3x3_s2,
    inception_3a_1x1,
    inception_3a_3x3_reduce,
    inception_3a_3x3,
    inception_3a_5x5_reduce,
    inception_3a_5x5,
    inception_3a_pool,
    inception_3a_pool_proj,
    inception_3a_output,
    inception_3b_1x1,
    inception_3b_3x3_reduce,
    inception_3b_3x3,
    inception_3b_5x5_reduce,
    inception_3b_5x5,
    inception_3b_pool,
    inception_3b_pool_proj,
    inception_3b_output,
    pool3_3x3_s2,
    inception_4a_1x1,
    inception_4a_3x3_reduce,
    inception_4a_3x3,
    inception_4a_5x5_reduce,
    inception_4a_5x5,
    inception_4a_pool,
    inception_4a_pool_proj,
    inception_4a_output,
    inception_4b_1x1,
    inception_4b_3x3_reduce,
    inception_4b_3x3,
    inception_4b_5x5_reduce,
    inception_4b_5x5,
    inception_4b_pool,
    inception_4b_pool_proj,
    inception_4b_output,
    inception_4c_1x1,
    inception_4c_3x3_reduce,
    inception_4c_3x3,
    inception_4c_5x5_reduce,
    inception_4c_5x5,
    inception_4c_pool,
    inception_4c_pool_proj,
    inception_4c_output,
    inception_4d_1x1,
    inception_4d_3x3_reduce,
    inception_4d_3x3,
    inception_4d_5x5_reduce,
    inception_4d_5x5,
    inception_4d_pool,
    inception_4d_pool_proj,
    inception_4d_output,
    inception_4e_1x1,
    inception_4e_3x3_reduce,
    inception_4e_3x3,
    inception_4e_5x5_reduce,
    inception_4e_5x5,
    inception_4e_pool,
    inception_4e_pool_proj,
    inception_4e_output,
    pool4_3x3_s2,
    inception_5a_1x1,
    inception_5a_3x3_reduce,
    inception_5a_3x3,
    inception_5a_5x5_reduce,
    inception_5a_5x5,
    inception_5a_pool,
    inception_5a_pool_proj,
    inception_5a_output,
    inception_5b_1x1,
    inception_5b_3x3_reduce,
    inception_5b_3x3,
    inception_5b_5x5_reduce,
    inception_5b_5x5,
    inception_5b_pool,
    inception_5b_pool_proj,
    inception_5b_output,
    pool5_7x7_s1,
    loss3_classifier,
    softmax,
    output,
    last_workflow_item = output
};

enum  workflow_item_factor {
    mean_factor,
    conv1_7x7_s2_weights,
    conv1_7x7_s2_biases,
    conv2_3x3_reduce_weights,
    conv2_3x3_reduce_biases,
    conv2_3x3_weights,
    conv2_3x3_biases,
    inception_3a_1x1_weights,
    inception_3a_1x1_biases,
    inception_3a_3x3_reduce_weights,
    inception_3a_3x3_reduce_biases,
    inception_3a_3x3_weights,
    inception_3a_3x3_biases,
    inception_3a_5x5_reduce_weights,
    inception_3a_5x5_reduce_biases,
    inception_3a_5x5_weights,
    inception_3a_5x5_biases,
    inception_3a_pool_proj_weights,
    inception_3a_pool_proj_biases,
    inception_3b_1x1_weights,
    inception_3b_1x1_biases,
    inception_3b_3x3_reduce_weights,
    inception_3b_3x3_reduce_biases,
    inception_3b_3x3_weights,
    inception_3b_3x3_biases,
    inception_3b_5x5_reduce_weights,
    inception_3b_5x5_reduce_biases,
    inception_3b_5x5_weights,
    inception_3b_5x5_biases,
    inception_3b_pool_proj_weights,
    inception_3b_pool_proj_biases,
    inception_4a_1x1_weights,
    inception_4a_1x1_biases,
    inception_4a_3x3_reduce_weights,
    inception_4a_3x3_reduce_biases,
    inception_4a_3x3_weights,
    inception_4a_3x3_biases,
    inception_4a_5x5_reduce_weights,
    inception_4a_5x5_reduce_biases,
    inception_4a_5x5_weights,
    inception_4a_5x5_biases,
    inception_4a_pool_proj_weights,
    inception_4a_pool_proj_biases,
    inception_4b_1x1_weights,
    inception_4b_1x1_biases,
    inception_4b_3x3_reduce_weights,
    inception_4b_3x3_reduce_biases,
    inception_4b_3x3_weights,
    inception_4b_3x3_biases,
    inception_4b_5x5_reduce_weights,
    inception_4b_5x5_reduce_biases,
    inception_4b_5x5_weights,
    inception_4b_5x5_biases,
    inception_4b_pool_proj_weights,
    inception_4b_pool_proj_biases,
    inception_4c_1x1_weights,
    inception_4c_1x1_biases,
    inception_4c_3x3_reduce_weights,
    inception_4c_3x3_reduce_biases,
    inception_4c_3x3_weights,
    inception_4c_3x3_biases,
    inception_4c_5x5_reduce_weights,
    inception_4c_5x5_reduce_biases,
    inception_4c_5x5_weights,
    inception_4c_5x5_biases,
    inception_4c_pool_proj_weights,
    inception_4c_pool_proj_biases,
    inception_4d_1x1_weights,
    inception_4d_1x1_biases,
    inception_4d_3x3_reduce_weights,
    inception_4d_3x3_reduce_biases,
    inception_4d_3x3_weights,
    inception_4d_3x3_biases,
    inception_4d_5x5_reduce_weights,
    inception_4d_5x5_reduce_biases,
    inception_4d_5x5_weights,
    inception_4d_5x5_biases,
    inception_4d_pool_proj_weights,
    inception_4d_pool_proj_biases,
    inception_4e_1x1_weights,
    inception_4e_1x1_biases,
    inception_4e_3x3_reduce_weights,
    inception_4e_3x3_reduce_biases,
    inception_4e_3x3_weights,
    inception_4e_3x3_biases,
    inception_4e_5x5_reduce_weights,
    inception_4e_5x5_reduce_biases,
    inception_4e_5x5_weights,
    inception_4e_5x5_biases,
    inception_4e_pool_proj_weights,
    inception_4e_pool_proj_biases,
    inception_5a_1x1_weights,
    inception_5a_1x1_biases,
    inception_5a_3x3_reduce_weights,
    inception_5a_3x3_reduce_biases,
    inception_5a_3x3_weights,
    inception_5a_3x3_biases,
    inception_5a_5x5_reduce_weights,
    inception_5a_5x5_reduce_biases,
    inception_5a_5x5_weights,
    inception_5a_5x5_biases,
    inception_5a_pool_proj_weights,
    inception_5a_pool_proj_biases,
    inception_5b_1x1_weights,
    inception_5b_1x1_biases,
    inception_5b_3x3_reduce_weights,
    inception_5b_3x3_reduce_biases,
    inception_5b_3x3_weights,
    inception_5b_3x3_biases,
    inception_5b_5x5_reduce_weights,
    inception_5b_5x5_reduce_biases,
    inception_5b_5x5_weights,
    inception_5b_5x5_biases,
    inception_5b_pool_proj_weights,
    inception_5b_pool_proj_biases,
    loss3_classifier_weights,
    loss3_classifier_biases,
    last_factor = loss3_classifier_biases
};

class workflow_builder_googlenet_float : public workflow_builder_base
{

public:

    workflow_builder_googlenet_float() : workflow_builder_base(229) {
        RGB_order = false;
        image_process = fi::resize_image_to_square;

        for(auto& wi : workflow_item) wi = nullptr;
        for(auto& wif : workflow_item_factor) wif = nullptr;

        try {
            read_file_to_vector(labels, "weights_googlenet/names.txt", false);
            read_file_to_vector(wwids, "weights_googlenet/wwids.txt", false);
        }
        catch (std::runtime_error &e) {
            error_ = e.what();
        }
    }

    bool is_valid() { return error_.empty(); }

    virtual NN_WORKLOAD_DATA_TYPE* get_input_formats() { return in_formats; }
    virtual NN_WORKLOAD_DATA_TYPE* get_output_formats() { return out_formats; }

private:
    std::string error_;

    // pointers to successive workflow parts
    nn_workflow_item_t        *workflow_item[last_workflow_item+1];

    // pointers to nn_datas containing weights and biases;
    nn::data<float>           *workflow_item_factor[last_factor+1];

    nn_workflow_t           *workflow = nullptr;
    nn_device_interface_0_t *di = nullptr;

public:

    void cleanup(){
        if (!is_valid()) throw std::runtime_error(error_);

        /* ****************************************************************************************** */
        /* Cleanup in memory                                                                          */
        /* ****************************************************************************************** */
        std::cout
            << "Cleanup in memory"
            << std::endl
            << "========================================================"
            << std::endl;

        for(auto& wl : workflow_item)
                di->workflow_item_delete_function(wl);

        di->workflow_delete_function(workflow);

        for(auto& wb : workflow_item_factor)
            if(wb!=nullptr) delete wb;
    }

    virtual nn_workflow_t *init_workflow(nn_device_interface_0_t *di) {

        if (!is_valid()) throw std::runtime_error(error_);

        this->di = di;

        std::cout
            << "--------------------------------------------------------"
            << std::endl
            << "Loading weights and biases"
            << std::endl << std::endl;

        // Load weights and biases
        auto load_biases_or_weights = [](std::string wb_file_name) {
            nn::data<float> *wb_pointer = nn_data_load_from_file_time_measure(wb_file_name);
            if (wb_pointer == nullptr) {
                std::cerr << "Can't load " << wb_file_name << std::endl;
                throw;
            }

            return wb_pointer;
        };

        try {
            workflow_item_factor[mean_factor] = load_biases_or_weights("weights_googlenet/googlenet_mean.nnd");
            workflow_item_factor[conv1_7x7_s2_weights] = load_biases_or_weights("weights_googlenet/conv1_7x7_s2.nnd");
            workflow_item_factor[conv1_7x7_s2_biases] = load_biases_or_weights("weights_googlenet/conv1_7x7_s2_bias.nnd");
            workflow_item_factor[conv2_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/conv2_3x3_reduce.nnd");
            workflow_item_factor[conv2_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/conv2_3x3_reduce_bias.nnd");
            workflow_item_factor[conv2_3x3_weights] = load_biases_or_weights("weights_googlenet/conv2_3x3.nnd");
            workflow_item_factor[conv2_3x3_biases] = load_biases_or_weights("weights_googlenet/conv2_3x3_bias.nnd");
            workflow_item_factor[inception_3a_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_3a_1x1.nnd");
            workflow_item_factor[inception_3a_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_3a_1x1_bias.nnd");
            workflow_item_factor[inception_3a_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_3a_3x3_reduce.nnd");
            workflow_item_factor[inception_3a_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_3a_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_3a_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_3a_3x3.nnd");
            workflow_item_factor[inception_3a_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_3a_3x3_bias.nnd");
            workflow_item_factor[inception_3a_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_3a_5x5_reduce.nnd");
            workflow_item_factor[inception_3a_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_3a_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_3a_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_3a_5x5.nnd");
            workflow_item_factor[inception_3a_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_3a_5x5_bias.nnd");
            workflow_item_factor[inception_3a_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_3a_pool_proj.nnd");
            workflow_item_factor[inception_3a_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_3a_pool_proj_bias.nnd");
            workflow_item_factor[inception_3b_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_3b_1x1.nnd");
            workflow_item_factor[inception_3b_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_3b_1x1_bias.nnd");
            workflow_item_factor[inception_3b_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_3b_3x3_reduce.nnd");
            workflow_item_factor[inception_3b_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_3b_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_3b_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_3b_3x3.nnd");
            workflow_item_factor[inception_3b_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_3b_3x3_bias.nnd");
            workflow_item_factor[inception_3b_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_3b_5x5_reduce.nnd");
            workflow_item_factor[inception_3b_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_3b_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_3b_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_3b_5x5.nnd");
            workflow_item_factor[inception_3b_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_3b_5x5_bias.nnd");
            workflow_item_factor[inception_3b_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_3b_pool_proj.nnd");
            workflow_item_factor[inception_3b_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_3b_pool_proj_bias.nnd");
            workflow_item_factor[inception_4a_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_4a_1x1.nnd");
            workflow_item_factor[inception_4a_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_4a_1x1_bias.nnd");
            workflow_item_factor[inception_4a_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4a_3x3_reduce.nnd");
            workflow_item_factor[inception_4a_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4a_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_4a_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_4a_3x3.nnd");
            workflow_item_factor[inception_4a_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_4a_3x3_bias.nnd");
            workflow_item_factor[inception_4a_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4a_5x5_reduce.nnd");
            workflow_item_factor[inception_4a_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4a_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_4a_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_4a_5x5.nnd");
            workflow_item_factor[inception_4a_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_4a_5x5_bias.nnd");
            workflow_item_factor[inception_4a_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_4a_pool_proj.nnd");
            workflow_item_factor[inception_4a_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_4a_pool_proj_bias.nnd");
            workflow_item_factor[inception_4b_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_4b_1x1.nnd");
            workflow_item_factor[inception_4b_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_4b_1x1_bias.nnd");
            workflow_item_factor[inception_4b_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4b_3x3_reduce.nnd");
            workflow_item_factor[inception_4b_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4b_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_4b_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_4b_3x3.nnd");
            workflow_item_factor[inception_4b_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_4b_3x3_bias.nnd");
            workflow_item_factor[inception_4b_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4b_5x5_reduce.nnd");
            workflow_item_factor[inception_4b_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4b_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_4b_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_4b_5x5.nnd");
            workflow_item_factor[inception_4b_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_4b_5x5_bias.nnd");
            workflow_item_factor[inception_4b_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_4b_pool_proj.nnd");
            workflow_item_factor[inception_4b_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_4b_pool_proj_bias.nnd");
            workflow_item_factor[inception_4c_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_4c_1x1.nnd");
            workflow_item_factor[inception_4c_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_4c_1x1_bias.nnd");
            workflow_item_factor[inception_4c_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4c_3x3_reduce.nnd");
            workflow_item_factor[inception_4c_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4c_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_4c_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_4c_3x3.nnd");
            workflow_item_factor[inception_4c_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_4c_3x3_bias.nnd");
            workflow_item_factor[inception_4c_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4c_5x5_reduce.nnd");
            workflow_item_factor[inception_4c_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4c_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_4c_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_4c_5x5.nnd");
            workflow_item_factor[inception_4c_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_4c_5x5_bias.nnd");
            workflow_item_factor[inception_4c_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_4c_pool_proj.nnd");
            workflow_item_factor[inception_4c_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_4c_pool_proj_bias.nnd");
            workflow_item_factor[inception_4d_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_4d_1x1.nnd");
            workflow_item_factor[inception_4d_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_4d_1x1_bias.nnd");
            workflow_item_factor[inception_4d_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4d_3x3_reduce.nnd");
            workflow_item_factor[inception_4d_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4d_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_4d_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_4d_3x3.nnd");
            workflow_item_factor[inception_4d_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_4d_3x3_bias.nnd");
            workflow_item_factor[inception_4d_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4d_5x5_reduce.nnd");
            workflow_item_factor[inception_4d_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4d_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_4d_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_4d_5x5.nnd");
            workflow_item_factor[inception_4d_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_4d_5x5_bias.nnd");
            workflow_item_factor[inception_4d_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_4d_pool_proj.nnd");
            workflow_item_factor[inception_4d_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_4d_pool_proj_bias.nnd");
            workflow_item_factor[inception_4e_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_4e_1x1.nnd");
            workflow_item_factor[inception_4e_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_4e_1x1_bias.nnd");
            workflow_item_factor[inception_4e_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4e_3x3_reduce.nnd");
            workflow_item_factor[inception_4e_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4e_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_4e_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_4e_3x3.nnd");
            workflow_item_factor[inception_4e_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_4e_3x3_bias.nnd");
            workflow_item_factor[inception_4e_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_4e_5x5_reduce.nnd");
            workflow_item_factor[inception_4e_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_4e_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_4e_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_4e_5x5.nnd");
            workflow_item_factor[inception_4e_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_4e_5x5_bias.nnd");
            workflow_item_factor[inception_4e_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_4e_pool_proj.nnd");
            workflow_item_factor[inception_4e_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_4e_pool_proj_bias.nnd");
            workflow_item_factor[inception_5a_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_5a_1x1.nnd");
            workflow_item_factor[inception_5a_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_5a_1x1_bias.nnd");
            workflow_item_factor[inception_5a_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_5a_3x3_reduce.nnd");
            workflow_item_factor[inception_5a_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_5a_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_5a_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_5a_3x3.nnd");
            workflow_item_factor[inception_5a_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_5a_3x3_bias.nnd");
            workflow_item_factor[inception_5a_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_5a_5x5_reduce.nnd");
            workflow_item_factor[inception_5a_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_5a_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_5a_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_5a_5x5.nnd");
            workflow_item_factor[inception_5a_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_5a_5x5_bias.nnd");
            workflow_item_factor[inception_5a_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_5a_pool_proj.nnd");
            workflow_item_factor[inception_5a_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_5a_pool_proj_bias.nnd");
            workflow_item_factor[inception_5b_1x1_weights] = load_biases_or_weights("weights_googlenet/inception_5b_1x1.nnd");
            workflow_item_factor[inception_5b_1x1_biases] = load_biases_or_weights("weights_googlenet/inception_5b_1x1_bias.nnd");
            workflow_item_factor[inception_5b_3x3_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_5b_3x3_reduce.nnd");
            workflow_item_factor[inception_5b_3x3_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_5b_3x3_reduce_bias.nnd");
            workflow_item_factor[inception_5b_3x3_weights] = load_biases_or_weights("weights_googlenet/inception_5b_3x3.nnd");
            workflow_item_factor[inception_5b_3x3_biases] = load_biases_or_weights("weights_googlenet/inception_5b_3x3_bias.nnd");
            workflow_item_factor[inception_5b_5x5_reduce_weights] = load_biases_or_weights("weights_googlenet/inception_5b_5x5_reduce.nnd");
            workflow_item_factor[inception_5b_5x5_reduce_biases] = load_biases_or_weights("weights_googlenet/inception_5b_5x5_reduce_bias.nnd");
            workflow_item_factor[inception_5b_5x5_weights] = load_biases_or_weights("weights_googlenet/inception_5b_5x5.nnd");
            workflow_item_factor[inception_5b_5x5_biases] = load_biases_or_weights("weights_googlenet/inception_5b_5x5_bias.nnd");
            workflow_item_factor[inception_5b_pool_proj_weights] = load_biases_or_weights("weights_googlenet/inception_5b_pool_proj.nnd");
            workflow_item_factor[inception_5b_pool_proj_biases] = load_biases_or_weights("weights_googlenet/inception_5b_pool_proj_bias.nnd");
            workflow_item_factor[loss3_classifier_weights] = load_biases_or_weights("weights_googlenet/loss3_classifier.nnd");
            workflow_item_factor[loss3_classifier_biases] = load_biases_or_weights("weights_googlenet/loss3_classifier_bias.nnd");
        }
        catch (...) {
            return workflow;
        }

        std::cout
            << "--------------------------------------------------------" << std::endl
            << "Build of workflow" << std::endl;

        di->workflow_create_function(&workflow, 1, 1);

        // ------------------------------------------------------------------------------------------
        // STAGE 0 (input)
        //         output: 229x229x3
        {
            di->workflow_item_create_function(&workflow_item[input], 0, nullptr, 1);

            workflow_item[input]->type = NN_WORK_ITEM_TYPE_INPUT;
            workflow_item[input]->arguments.input.index = 0;
            workflow_item[input]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[input]->output_format[0].format_3d = { { img_size, img_size, 3 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[input], 0 };
            di->workflow_item_create_function(&workflow_item[mean], 1, &inputs_descriptor, 1);

            workflow_item[mean]->type = NN_WORK_ITEM_TYPE_ARITHMETIC;
            workflow_item[mean]->arguments.forward_arithmetic.factor = workflow_item_factor[mean_factor];
            workflow_item[mean]->arguments.forward_arithmetic.arithmetic_function = NN_ARITHMETIC_FUNCTION_SUBTRACTION;

            workflow_item[mean]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[mean]->output_format[0].format_3d = { { img_size, img_size, 3 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[mean], 0 };
            di->workflow_item_create_function(&workflow_item[conv1_7x7_s2], 1, &inputs_descriptor, 1);

            workflow_item[conv1_7x7_s2]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[conv1_7x7_s2]->name = "conv1_7x7_s2";

            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;

            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.weights = workflow_item_factor[conv1_7x7_s2_weights];
            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.biases = workflow_item_factor[conv1_7x7_s2_biases];

            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.stride[0] = 2;
            workflow_item[conv1_7x7_s2]->arguments.forward_convolution.stride[1] = 2;

            workflow_item[conv1_7x7_s2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[conv1_7x7_s2]->output_format[0].format_3d = { { 112, 112, 64 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[conv1_7x7_s2], 0 };
            di->workflow_item_create_function(&workflow_item[pool1_3x3_s2], 1, &inputs_descriptor, 1);

            workflow_item[pool1_3x3_s2]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[pool1_3x3_s2]->name = "pool1_3x3_s2";

            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.size[0] = 3;
            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.size[1] = 3;
            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.stride[0] = 2;
            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.stride[1] = 2;

            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.center_offset[0] = 0;
            workflow_item[pool1_3x3_s2]->arguments.forward_pooling.center_offset[1] = 0;

            workflow_item[pool1_3x3_s2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[pool1_3x3_s2]->output_format[0].format_3d = { { 56, 56, 64 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool1_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[pool1_norm1], 1, &inputs_descriptor, 1);

            workflow_item[pool1_norm1]->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            workflow_item[pool1_norm1]->name = "pool1_norm1";

            workflow_item[pool1_norm1]->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            workflow_item[pool1_norm1]->arguments.forward_normalization.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[pool1_norm1]->arguments.forward_normalization.normalization.k = 1; // in Krishevsky's article is 2
            workflow_item[pool1_norm1]->arguments.forward_normalization.normalization.n = 5;
            workflow_item[pool1_norm1]->arguments.forward_normalization.normalization.alpha = 0.0001f / 5; // in Krishevsky's paper is 1e-4,
                                                                                                           // but didn't write that sum of the squares
                                                                                                           // is divided by number of elements (n)
            workflow_item[pool1_norm1]->arguments.forward_normalization.normalization.beta = 0.75f;

            workflow_item[pool1_norm1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[pool1_norm1]->output_format[0].format_3d = { { 56, 56, 64 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool1_norm1], 0 };
            di->workflow_item_create_function(&workflow_item[conv2_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[conv2_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[conv2_3x3_reduce]->name = "conv2_3x3_reduce";

            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[conv2_3x3_reduce_weights];
            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[conv2_3x3_reduce_biases];

            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[conv2_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[conv2_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[conv2_3x3_reduce]->output_format[0].format_3d = { { 56, 56, 64 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[conv2_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[conv2_3x3], 1, &inputs_descriptor, 1);

            workflow_item[conv2_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[conv2_3x3]->name = "conv2_3x3";

            workflow_item[conv2_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[conv2_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[conv2_3x3]->arguments.forward_convolution.weights = workflow_item_factor[conv2_3x3_weights];
            workflow_item[conv2_3x3]->arguments.forward_convolution.biases = workflow_item_factor[conv2_3x3_biases];

            workflow_item[conv2_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[conv2_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[conv2_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[conv2_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[conv2_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[conv2_3x3]->output_format[0].format_3d = { { 56, 56, 192 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[conv2_3x3], 0 };
            di->workflow_item_create_function(&workflow_item[conv2_norm2], 1, &inputs_descriptor, 1);

            workflow_item[conv2_norm2]->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
            workflow_item[conv2_norm2]->name = "conv2_norm2";

            workflow_item[conv2_norm2]->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
            workflow_item[conv2_norm2]->arguments.forward_normalization.normalization.k = 1; // in Krishevsky's article is 2
            workflow_item[conv2_norm2]->arguments.forward_normalization.normalization.n = 5;
            workflow_item[conv2_norm2]->arguments.forward_normalization.normalization.alpha = 0.0001f / 5; // in Krishevsky's paper is 1e-4,
                                                                                                           // but didn't write that sum of the squares
                                                                                                           // is divided by number of elements (n)
            workflow_item[conv2_norm2]->arguments.forward_normalization.normalization.beta = 0.75f;

            workflow_item[conv2_norm2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[conv2_norm2]->output_format[0].format_3d = { { 56, 56, 192 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[conv2_norm2], 0 };
            di->workflow_item_create_function(&workflow_item[pool2_3x3_s2], 1, &inputs_descriptor, 1); // pooling

            workflow_item[pool2_3x3_s2]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[pool2_3x3_s2]->name = "pool2_3x3_s2";

            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.size[0] = 3;
            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.size[1] = 3;

            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.center_offset[0] = 0;
            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.center_offset[1] = 0;

            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.stride[0] = 2;
            workflow_item[pool2_3x3_s2]->arguments.forward_pooling.stride[1] = 2;

            workflow_item[pool2_3x3_s2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[pool2_3x3_s2]->output_format[0].format_3d = { { 28, 28, 192 } };
        }

        /* 3a inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool2_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3a_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_3a_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3a_1x1]->name = "inception_3a_1x1";

            workflow_item[inception_3a_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3a_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3a_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_3a_1x1_weights];
            workflow_item[inception_3a_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_3a_1x1_biases];

            workflow_item[inception_3a_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3a_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3a_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3a_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3a_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_1x1]->output_format[0].format_3d = { { 28, 28, 64 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool2_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3a_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_3a_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3a_3x3_reduce]->name = "inception_3a_3x3_reduce";

            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_3a_3x3_reduce_weights];
            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_3a_3x3_reduce_biases];

            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3a_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3a_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_3x3_reduce]->output_format[0].format_3d = { { 28, 28, 96 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3a_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3a_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_3a_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3a_3x3]->name = "inception_3a_3x3";

            workflow_item[inception_3a_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3a_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3a_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_3a_3x3_weights];
            workflow_item[inception_3a_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_3a_3x3_biases];

            workflow_item[inception_3a_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_3a_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_3a_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3a_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3a_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_3x3]->output_format[0].format_3d = { { 28, 28, 128 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool2_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3a_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_3a_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3a_5x5_reduce]->name = "inception_3a_5x5_reduce";

            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_3a_5x5_reduce_weights];
            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_3a_5x5_reduce_biases];

            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3a_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3a_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_5x5_reduce]->output_format[0].format_3d = { { 28, 28, 16 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3a_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3a_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_3a_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3a_5x5]->name = "inception_3a_5x5";

            workflow_item[inception_3a_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3a_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3a_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_3a_5x5_weights];
            workflow_item[inception_3a_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_3a_5x5_biases];

            workflow_item[inception_3a_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_3a_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_3a_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3a_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3a_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_5x5]->output_format[0].format_3d = { { 28, 28, 32 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool2_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3a_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_3a_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_3a_pool]->name = "inception_3a_pool";

            workflow_item[inception_3a_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_3a_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_3a_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_3a_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_3a_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_3a_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_3a_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_3a_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_3a_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_pool]->output_format[0].format_3d = { { 28, 28, 192 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3a_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3a_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_3a_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3a_pool_proj]->name = "inception_3a_pool_proj";

            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_3a_pool_proj_weights];
            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_3a_pool_proj_biases];

            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3a_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3a_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_pool_proj]->output_format[0].format_3d = { { 28, 28, 32 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_3a_1x1], 0 }, { workflow_item[inception_3a_3x3], 0 }, { workflow_item[inception_3a_5x5], 0 }, { workflow_item[inception_3a_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_3a_output], 4, inputs_descriptor, 1);

            workflow_item[inception_3a_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_3a_output]->name = "merge3a";
            workflow_item[inception_3a_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_3a_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3a_output]->output_format[0].format_3d = { { 28, 28, 256 } };
        }

        /* 3b inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3b_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_3b_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3b_1x1]->name = "inception_3b_1x1";

            workflow_item[inception_3b_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3b_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3b_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_3b_1x1_weights];
            workflow_item[inception_3b_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_3b_1x1_biases];

            workflow_item[inception_3b_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3b_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3b_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3b_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3b_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_1x1]->output_format[0].format_3d = { { 28, 28, 128 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3b_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_3b_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3b_3x3_reduce]->name = "inception_3b_3x3_reduce";

            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_3b_3x3_reduce_weights];
            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_3b_3x3_reduce_biases];

            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3b_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3b_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_3x3_reduce]->output_format[0].format_3d = { { 28, 28, 128 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3b_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3b_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_3b_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3b_3x3]->name = "inception_3b_3x3";

            workflow_item[inception_3b_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3b_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3b_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_3b_3x3_weights];
            workflow_item[inception_3b_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_3b_3x3_biases];

            workflow_item[inception_3b_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_3b_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_3b_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3b_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3b_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_3x3]->output_format[0].format_3d = { { 28, 28, 192 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3b_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_3b_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3b_5x5_reduce]->name = "inception_3b_5x5_reduce";

            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_3b_5x5_reduce_weights];
            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_3b_5x5_reduce_biases];

            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3b_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3b_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_5x5_reduce]->output_format[0].format_3d = { { 28, 28, 32 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3b_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3b_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_3b_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3b_5x5]->name = "inception_3b_5x5";

            workflow_item[inception_3b_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3b_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3b_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_3b_5x5_weights];
            workflow_item[inception_3b_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_3b_5x5_biases];

            workflow_item[inception_3b_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_3b_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_3b_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3b_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3b_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_5x5]->output_format[0].format_3d = { { 28, 28, 96 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3b_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_3b_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_3b_pool]->name = "inception_3b_pool";

            workflow_item[inception_3b_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_3b_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_3b_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_3b_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_3b_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_3b_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_3b_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_3b_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_3b_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_pool]->output_format[0].format_3d = { { 28, 28, 256 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3b_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_3b_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_3b_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_3b_pool_proj]->name = "inception_3b_pool_proj";

            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_3b_pool_proj_weights];
            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_3b_pool_proj_biases];

            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_3b_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_3b_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_pool_proj]->output_format[0].format_3d = { { 28, 28, 64 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_3b_1x1], 0 }, { workflow_item[inception_3b_3x3], 0 }, { workflow_item[inception_3b_5x5], 0 }, { workflow_item[inception_3b_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_3b_output], 4, inputs_descriptor, 1);

            workflow_item[inception_3b_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_3b_output]->name = "merge3b";
            workflow_item[inception_3b_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_3b_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_3b_output]->output_format[0].format_3d = { { 28, 28, 480 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_3b_output], 0 };
            di->workflow_item_create_function(&workflow_item[pool3_3x3_s2], 1, &inputs_descriptor, 1); // pooling

            workflow_item[pool3_3x3_s2]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[pool3_3x3_s2]->name = "pool3_3x3_s2";

            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.size[0] = 3;
            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.size[1] = 3;

            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.center_offset[0] = 0;
            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.center_offset[1] = 0;

            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.stride[0] = 2;
            workflow_item[pool3_3x3_s2]->arguments.forward_pooling.stride[1] = 2;

            workflow_item[pool3_3x3_s2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[pool3_3x3_s2]->output_format[0].format_3d = { { 14, 14, 480 } };
        }

        /* 4a inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool3_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4a_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_4a_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4a_1x1]->name = "inception_4a_1x1";

            workflow_item[inception_4a_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4a_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4a_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_4a_1x1_weights];
            workflow_item[inception_4a_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_4a_1x1_biases];

            workflow_item[inception_4a_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4a_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4a_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4a_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4a_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_1x1]->output_format[0].format_3d = { { 14, 14, 192 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool3_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4a_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4a_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4a_3x3_reduce]->name = "inception_4a_3x3_reduce";

            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4a_3x3_reduce_weights];
            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4a_3x3_reduce_biases];

            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4a_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4a_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_3x3_reduce]->output_format[0].format_3d = { { 14, 14, 96 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4a_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4a_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_4a_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4a_3x3]->name = "inception_4a_3x3";

            workflow_item[inception_4a_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4a_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4a_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_4a_3x3_weights];
            workflow_item[inception_4a_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_4a_3x3_biases];

            workflow_item[inception_4a_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_4a_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_4a_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4a_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4a_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_3x3]->output_format[0].format_3d = { { 14, 14, 208 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool3_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4a_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4a_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4a_5x5_reduce]->name = "inception_4a_5x5_reduce";

            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4a_5x5_reduce_weights];
            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4a_5x5_reduce_biases];

            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4a_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4a_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_5x5_reduce]->output_format[0].format_3d = { { 14, 14, 16 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4a_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4a_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_4a_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4a_5x5]->name = "inception_4a_5x5";

            workflow_item[inception_4a_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4a_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4a_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_4a_5x5_weights];
            workflow_item[inception_4a_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_4a_5x5_biases];

            workflow_item[inception_4a_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_4a_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_4a_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4a_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4a_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_5x5]->output_format[0].format_3d = { { 14, 14, 48 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool3_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4a_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_4a_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_4a_pool]->name = "inception_4a_pool";

            workflow_item[inception_4a_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_4a_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_4a_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_4a_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_4a_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_4a_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_4a_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_4a_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_4a_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_pool]->output_format[0].format_3d = { { 14, 14, 480 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4a_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4a_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_4a_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4a_pool_proj]->name = "inception_4a_pool_proj";

            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_4a_pool_proj_weights];
            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_4a_pool_proj_biases];

            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4a_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4a_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_pool_proj]->output_format[0].format_3d = { { 14, 14, 64 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_4a_1x1], 0 }, { workflow_item[inception_4a_3x3], 0 }, { workflow_item[inception_4a_5x5], 0 }, { workflow_item[inception_4a_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_4a_output], 4, inputs_descriptor, 1);

            workflow_item[inception_4a_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_4a_output]->name = "merge4a";
            workflow_item[inception_4a_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_4a_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4a_output]->output_format[0].format_3d = { { 14, 14, 512 } };
        }

        /* 4b inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4b_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_4b_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4b_1x1]->name = "inception_4b_1x1";

            workflow_item[inception_4b_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4b_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4b_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_4b_1x1_weights];
            workflow_item[inception_4b_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_4b_1x1_biases];

            workflow_item[inception_4b_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4b_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4b_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4b_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4b_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_1x1]->output_format[0].format_3d = { { 14, 14, 160 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4b_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4b_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4b_3x3_reduce]->name = "inception_4b_3x3_reduce";

            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4b_3x3_reduce_weights];
            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4b_3x3_reduce_biases];

            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4b_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4b_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_3x3_reduce]->output_format[0].format_3d = { { 14, 14, 112 } };
        }


        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4b_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4b_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_4b_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4b_3x3]->name = "inception_4b_3x3";

            workflow_item[inception_4b_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4b_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4b_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_4b_3x3_weights];
            workflow_item[inception_4b_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_4b_3x3_biases];

            workflow_item[inception_4b_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_4b_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_4b_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4b_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4b_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_3x3]->output_format[0].format_3d = { { 14, 14, 224 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4b_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4b_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4b_5x5_reduce]->name = "inception_4b_5x5_reduce";

            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4b_5x5_reduce_weights];
            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4b_5x5_reduce_biases];

            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4b_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4b_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_5x5_reduce]->output_format[0].format_3d = { { 14, 14, 24 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4b_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4b_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_4b_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4b_5x5]->name = "inception_4b_5x5";

            workflow_item[inception_4b_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4b_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4b_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_4b_5x5_weights];
            workflow_item[inception_4b_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_4b_5x5_biases];

            workflow_item[inception_4b_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_4b_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_4b_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4b_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4b_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_5x5]->output_format[0].format_3d = { { 14, 14, 64 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4b_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_4b_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_4b_pool]->name = "inception_4b_pool";

            workflow_item[inception_4b_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_4b_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_4b_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_4b_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_4b_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_4b_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_4b_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_4b_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_4b_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_pool]->output_format[0].format_3d = { { 14, 14, 512 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4b_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4b_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_4b_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4b_pool_proj]->name = "inception_4b_pool_proj";

            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_4b_pool_proj_weights];
            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_4b_pool_proj_biases];

            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4b_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4b_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_pool_proj]->output_format[0].format_3d = { { 14, 14, 64 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_4b_1x1], 0 }, { workflow_item[inception_4b_3x3], 0 }, { workflow_item[inception_4b_5x5], 0 }, { workflow_item[inception_4b_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_4b_output], 4, inputs_descriptor, 1);

            workflow_item[inception_4b_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_4b_output]->name = "merge4b";
            workflow_item[inception_4b_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_4b_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4b_output]->output_format[0].format_3d = { { 14, 14, 512 } };
        }

        /* 4c inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4b_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4c_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_4c_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4c_1x1]->name = "inception_4c_1x1";

            workflow_item[inception_4c_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4c_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4c_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_4c_1x1_weights];
            workflow_item[inception_4c_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_4c_1x1_biases];

            workflow_item[inception_4c_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4c_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4c_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4c_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4c_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_1x1]->output_format[0].format_3d = { { 14, 14, 128 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4b_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4c_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4c_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4c_3x3_reduce]->name = "inception_4c_3x3_reduce";

            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4c_3x3_reduce_weights];
            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4c_3x3_reduce_biases];

            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4c_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4c_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_3x3_reduce]->output_format[0].format_3d = { { 14, 14, 128 } };
        }


        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4c_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4c_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_4c_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4c_3x3]->name = "inception_4c_3x3";

            workflow_item[inception_4c_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4c_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4c_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_4c_3x3_weights];
            workflow_item[inception_4c_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_4c_3x3_biases];

            workflow_item[inception_4c_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_4c_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_4c_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4c_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4c_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_3x3]->output_format[0].format_3d = { { 14, 14, 256 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4b_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4c_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4c_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4c_5x5_reduce]->name = "inception_4c_5x5_reduce";

            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4c_5x5_reduce_weights];
            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4c_5x5_reduce_biases];

            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4c_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4c_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_5x5_reduce]->output_format[0].format_3d = { { 14, 14, 24 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4c_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4c_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_4c_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4c_5x5]->name = "inception_4c_5x5";

            workflow_item[inception_4c_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4c_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4c_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_4c_5x5_weights];
            workflow_item[inception_4c_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_4c_5x5_biases];

            workflow_item[inception_4c_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_4c_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_4c_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4c_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4c_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_5x5]->output_format[0].format_3d = { { 14, 14, 64 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4b_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4c_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_4c_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_4c_pool]->name = "inception_4c_pool";

            workflow_item[inception_4c_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_4c_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_4c_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_4c_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_4c_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_4c_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_4c_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_4c_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_4c_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_pool]->output_format[0].format_3d = { { 14, 14, 512 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4c_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4c_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_4c_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4c_pool_proj]->name = "inception_4c_pool_proj";

            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_4c_pool_proj_weights];
            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_4c_pool_proj_biases];

            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4c_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4c_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_pool_proj]->output_format[0].format_3d = { { 14, 14, 64 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_4c_1x1], 0 }, { workflow_item[inception_4c_3x3], 0 }, { workflow_item[inception_4c_5x5], 0 }, { workflow_item[inception_4c_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_4c_output], 4, inputs_descriptor, 1);

            workflow_item[inception_4c_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_4c_output]->name = "merge4c";
            workflow_item[inception_4c_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_4c_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4c_output]->output_format[0].format_3d = { { 14, 14, 512 } };
        }

        /* 4d inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4c_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4d_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_4d_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4d_1x1]->name = "inception_4d_1x1";

            workflow_item[inception_4d_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4d_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4d_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_4d_1x1_weights];
            workflow_item[inception_4d_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_4d_1x1_biases];

            workflow_item[inception_4d_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4d_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4d_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4d_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4d_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_1x1]->output_format[0].format_3d = { { 14, 14, 112 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4c_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4d_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4d_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4d_3x3_reduce]->name = "inception_4d_3x3_reduce";

            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4d_3x3_reduce_weights];
            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4d_3x3_reduce_biases];

            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4d_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4d_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_3x3_reduce]->output_format[0].format_3d = { { 14, 14, 144 } };
        }


        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4d_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4d_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_4d_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4d_3x3]->name = "inception_4d_3x3";

            workflow_item[inception_4d_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4d_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4d_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_4d_3x3_weights];
            workflow_item[inception_4d_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_4d_3x3_biases];

            workflow_item[inception_4d_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_4d_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_4d_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4d_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4d_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_3x3]->output_format[0].format_3d = { { 14, 14, 288 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4c_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4d_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4d_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4d_5x5_reduce]->name = "inception_4d_5x5_reduce";

            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4d_5x5_reduce_weights];
            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4d_5x5_reduce_biases];

            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4d_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4d_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_5x5_reduce]->output_format[0].format_3d = { { 14, 14, 32 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4d_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4d_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_4d_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4d_5x5]->name = "inception_4d_5x5";

            workflow_item[inception_4d_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4d_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4d_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_4d_5x5_weights];
            workflow_item[inception_4d_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_4d_5x5_biases];

            workflow_item[inception_4d_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_4d_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_4d_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4d_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4d_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_5x5]->output_format[0].format_3d = { { 14, 14, 64 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4c_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4d_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_4d_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_4d_pool]->name = "inception_4d_pool";

            workflow_item[inception_4d_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_4d_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_4d_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_4d_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_4d_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_4d_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_4d_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_4d_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_4d_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_pool]->output_format[0].format_3d = { { 14, 14, 512 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4d_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4d_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_4d_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4d_pool_proj]->name = "inception_4d_pool_proj";

            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_4d_pool_proj_weights];
            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_4d_pool_proj_biases];

            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4d_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4d_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_pool_proj]->output_format[0].format_3d = { { 14, 14, 64 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_4d_1x1], 0 }, { workflow_item[inception_4d_3x3], 0 }, { workflow_item[inception_4d_5x5], 0 }, { workflow_item[inception_4d_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_4d_output], 4, inputs_descriptor, 1);

            workflow_item[inception_4d_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_4d_output]->name = "merge4d";
            workflow_item[inception_4d_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_4d_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4d_output]->output_format[0].format_3d = { { 14, 14, 528 } };
        }

        /* 4e inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4d_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4e_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_4e_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4e_1x1]->name = "inception_4e_1x1";

            workflow_item[inception_4e_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4e_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4e_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_4e_1x1_weights];
            workflow_item[inception_4e_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_4e_1x1_biases];

            workflow_item[inception_4e_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4e_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4e_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4e_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4e_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_1x1]->output_format[0].format_3d = { { 14, 14, 256 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4d_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4e_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4e_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4e_3x3_reduce]->name = "inception_4e_3x3_reduce";

            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4e_3x3_reduce_weights];
            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4e_3x3_reduce_biases];

            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4e_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4e_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_3x3_reduce]->output_format[0].format_3d = { { 14, 14, 160 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4e_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4e_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_4e_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4e_3x3]->name = "inception_4e_3x3";

            workflow_item[inception_4e_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4e_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4e_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_4e_3x3_weights];
            workflow_item[inception_4e_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_4e_3x3_biases];

            workflow_item[inception_4e_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_4e_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_4e_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4e_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4e_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_3x3]->output_format[0].format_3d = { { 14, 14, 320 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4d_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4e_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_4e_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4e_5x5_reduce]->name = "inception_4e_5x5_reduce";

            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_4e_5x5_reduce_weights];
            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_4e_5x5_reduce_biases];

            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4e_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4e_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_5x5_reduce]->output_format[0].format_3d = { { 14, 14, 32 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4e_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4e_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_4e_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4e_5x5]->name = "inception_4e_5x5";

            workflow_item[inception_4e_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4e_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4e_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_4e_5x5_weights];
            workflow_item[inception_4e_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_4e_5x5_biases];

            workflow_item[inception_4e_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_4e_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_4e_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4e_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4e_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_5x5]->output_format[0].format_3d = { { 14, 14, 128 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4d_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4e_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_4e_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_4e_pool]->name = "inception_4e_pool";

            workflow_item[inception_4e_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_4e_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_4e_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_4e_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_4e_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_4e_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_4e_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_4e_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_4e_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_pool]->output_format[0].format_3d = { { 14, 14, 528 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4e_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_4e_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_4e_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_4e_pool_proj]->name = "inception_4e_pool_proj";

            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_4e_pool_proj_weights];
            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_4e_pool_proj_biases];

            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_4e_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_4e_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_pool_proj]->output_format[0].format_3d = { { 14, 14, 128 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_4e_1x1], 0 }, { workflow_item[inception_4e_3x3], 0 }, { workflow_item[inception_4e_5x5], 0 }, { workflow_item[inception_4e_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_4e_output], 4, inputs_descriptor, 1);

            workflow_item[inception_4e_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_4e_output]->name = "merge4e";
            workflow_item[inception_4e_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_4e_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_4e_output]->output_format[0].format_3d = { { 14, 14, 832 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_4e_output], 0 };
            di->workflow_item_create_function(&workflow_item[pool4_3x3_s2], 1, &inputs_descriptor, 1); // pooling

            workflow_item[pool4_3x3_s2]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[pool4_3x3_s2]->name = "pool4_3x3_s2";

            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.size[0] = 3;
            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.size[1] = 3;

            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.center_offset[0] = 0;
            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.center_offset[1] = 0;

            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.stride[0] = 2;
            workflow_item[pool4_3x3_s2]->arguments.forward_pooling.stride[1] = 2;

            workflow_item[pool4_3x3_s2]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[pool4_3x3_s2]->output_format[0].format_3d = { { 7, 7, 832 } };
        }

        /* 5a inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool4_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5a_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_5a_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5a_1x1]->name = "inception_5a_1x1";

            workflow_item[inception_5a_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5a_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5a_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_5a_1x1_weights];
            workflow_item[inception_5a_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_5a_1x1_biases];

            workflow_item[inception_5a_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5a_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5a_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5a_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5a_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_1x1]->output_format[0].format_3d = { { 7, 7, 256 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool4_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5a_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_5a_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5a_3x3_reduce]->name = "inception_5a_3x3_reduce";

            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_5a_3x3_reduce_weights];
            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_5a_3x3_reduce_biases];

            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5a_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5a_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_3x3_reduce]->output_format[0].format_3d = { { 7, 7, 160 } };
        }


        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5a_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5a_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_5a_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5a_3x3]->name = "inception_5a_3x3";

            workflow_item[inception_5a_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5a_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5a_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_5a_3x3_weights];
            workflow_item[inception_5a_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_5a_3x3_biases];

            workflow_item[inception_5a_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_5a_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_5a_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5a_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5a_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_3x3]->output_format[0].format_3d = { { 7, 7, 320 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool4_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5a_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_5a_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5a_5x5_reduce]->name = "inception_5a_5x5_reduce";

            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_5a_5x5_reduce_weights];
            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_5a_5x5_reduce_biases];

            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5a_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5a_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_5x5_reduce]->output_format[0].format_3d = { { 7, 7, 32 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5a_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5a_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_5a_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5a_5x5]->name = "inception_5a_5x5";

            workflow_item[inception_5a_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5a_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5a_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_5a_5x5_weights];
            workflow_item[inception_5a_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_5a_5x5_biases];

            workflow_item[inception_5a_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_5a_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_5a_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5a_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5a_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_5x5]->output_format[0].format_3d = { { 7, 7, 128 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool4_3x3_s2], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5a_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_5a_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_5a_pool]->name = "inception_5a_pool";

            workflow_item[inception_5a_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_5a_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_5a_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_5a_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_5a_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_5a_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_5a_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_5a_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_5a_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_pool]->output_format[0].format_3d = { { 7, 7, 832 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5a_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5a_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_5a_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5a_pool_proj]->name = "inception_5a_pool_proj";

            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_5a_pool_proj_weights];
            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_5a_pool_proj_biases];

            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5a_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5a_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_pool_proj]->output_format[0].format_3d = { { 7, 7, 128 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_5a_1x1], 0 }, { workflow_item[inception_5a_3x3], 0 }, { workflow_item[inception_5a_5x5], 0 }, { workflow_item[inception_5a_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_5a_output], 4, inputs_descriptor, 1);

            workflow_item[inception_5a_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_5a_output]->name = "merge5a";
            workflow_item[inception_5a_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_5a_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5a_output]->output_format[0].format_3d = { { 7, 7, 832 } };
        }

        /* 5b inception */
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5b_1x1], 1, &inputs_descriptor, 1);

            workflow_item[inception_5b_1x1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5b_1x1]->name = "inception_5b_1x1";

            workflow_item[inception_5b_1x1]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5b_1x1]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5b_1x1]->arguments.forward_convolution.weights = workflow_item_factor[inception_5b_1x1_weights];
            workflow_item[inception_5b_1x1]->arguments.forward_convolution.biases = workflow_item_factor[inception_5b_1x1_biases];

            workflow_item[inception_5b_1x1]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5b_1x1]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5b_1x1]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5b_1x1]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5b_1x1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_1x1]->output_format[0].format_3d = { { 7, 7, 384 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5b_3x3_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_5b_3x3_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5b_3x3_reduce]->name = "inception_5b_3x3_reduce";

            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_5b_3x3_reduce_weights];
            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_5b_3x3_reduce_biases];

            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5b_3x3_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5b_3x3_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_3x3_reduce]->output_format[0].format_3d = { { 7, 7, 192 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5b_3x3_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5b_3x3], 1, &inputs_descriptor, 1);

            workflow_item[inception_5b_3x3]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5b_3x3]->name = "inception_5b_3x3";

            workflow_item[inception_5b_3x3]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5b_3x3]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5b_3x3]->arguments.forward_convolution.weights = workflow_item_factor[inception_5b_3x3_weights];
            workflow_item[inception_5b_3x3]->arguments.forward_convolution.biases = workflow_item_factor[inception_5b_3x3_biases];

            workflow_item[inception_5b_3x3]->arguments.forward_convolution.center_offset[0] = 1;
            workflow_item[inception_5b_3x3]->arguments.forward_convolution.center_offset[1] = 1;

            workflow_item[inception_5b_3x3]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5b_3x3]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5b_3x3]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_3x3]->output_format[0].format_3d = { { 7, 7, 384 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5b_5x5_reduce], 1, &inputs_descriptor, 1);

            workflow_item[inception_5b_5x5_reduce]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5b_5x5_reduce]->name = "inception_5b_5x5_reduce";

            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.weights = workflow_item_factor[inception_5b_5x5_reduce_weights];
            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.biases = workflow_item_factor[inception_5b_5x5_reduce_biases];

            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5b_5x5_reduce]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5b_5x5_reduce]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_5x5_reduce]->output_format[0].format_3d = { { 7, 7, 48 } };
        }

        // padding: 2
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5b_5x5_reduce], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5b_5x5], 1, &inputs_descriptor, 1);

            workflow_item[inception_5b_5x5]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5b_5x5]->name = "inception_5b_5x5";

            workflow_item[inception_5b_5x5]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5b_5x5]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5b_5x5]->arguments.forward_convolution.weights = workflow_item_factor[inception_5b_5x5_weights];
            workflow_item[inception_5b_5x5]->arguments.forward_convolution.biases = workflow_item_factor[inception_5b_5x5_biases];

            workflow_item[inception_5b_5x5]->arguments.forward_convolution.center_offset[0] = 2;
            workflow_item[inception_5b_5x5]->arguments.forward_convolution.center_offset[1] = 2;

            workflow_item[inception_5b_5x5]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5b_5x5]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5b_5x5]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_5x5]->output_format[0].format_3d = { { 7, 7, 128 } };
        }

        // padding: 1
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5a_output], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5b_pool], 1, &inputs_descriptor, 1); // pooling

            workflow_item[inception_5b_pool]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[inception_5b_pool]->name = "inception_5b_pool";

            workflow_item[inception_5b_pool]->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;
            workflow_item[inception_5b_pool]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[inception_5b_pool]->arguments.forward_pooling.size[0] = 3;
            workflow_item[inception_5b_pool]->arguments.forward_pooling.size[1] = 3;

            workflow_item[inception_5b_pool]->arguments.forward_pooling.center_offset[0] = 1;
            workflow_item[inception_5b_pool]->arguments.forward_pooling.center_offset[1] = 1;

            workflow_item[inception_5b_pool]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[inception_5b_pool]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[inception_5b_pool]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_pool]->output_format[0].format_3d = { { 7, 7, 832 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5b_pool], 0 };
            di->workflow_item_create_function(&workflow_item[inception_5b_pool_proj], 1, &inputs_descriptor, 1);

            workflow_item[inception_5b_pool_proj]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
            workflow_item[inception_5b_pool_proj]->name = "inception_5b_pool_proj";

            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.activation.function = NN_ACTIVATION_FUNCTION_RELU;
            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.padding = NN_PADDING_MODE_DATA_OR_ZERO;

            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.weights = workflow_item_factor[inception_5b_pool_proj_weights];
            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.biases = workflow_item_factor[inception_5b_pool_proj_biases];

            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.center_offset[0] = 0;
            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.center_offset[1] = 0;

            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.stride[0] = 1;
            workflow_item[inception_5b_pool_proj]->arguments.forward_convolution.stride[1] = 1;

            workflow_item[inception_5b_pool_proj]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_pool_proj]->output_format[0].format_3d = { { 7, 7, 128 } };
        }

        // merge
        {
            nn_workflow_use_descriptor_t inputs_descriptor[] = { { workflow_item[inception_5b_1x1], 0 }, { workflow_item[inception_5b_3x3], 0 }, { workflow_item[inception_5b_5x5], 0 }, { workflow_item[inception_5b_pool_proj], 0 } };
            di->workflow_item_create_function(&workflow_item[inception_5b_output], 4, inputs_descriptor, 1);

            workflow_item[inception_5b_output]->type = NN_WORK_ITEM_TYPE_MERGE;
            workflow_item[inception_5b_output]->name = "merge5b";
            workflow_item[inception_5b_output]->arguments.forward_merge.axis = 2; // value 2 for z-axis

            workflow_item[inception_5b_output]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[inception_5b_output]->output_format[0].format_3d = { { 7, 7, 1024 } };
        }

        // Average pool
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[inception_5b_output], 0 };
            di->workflow_item_create_function(&workflow_item[pool5_7x7_s1], 1, &inputs_descriptor, 1); // pooling

            workflow_item[pool5_7x7_s1]->type = NN_WORK_ITEM_TYPE_POOLING;
            workflow_item[pool5_7x7_s1]->name = "pool5_7x7_s1";

            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.mode = NN_POOLING_MODE_AVERAGE;
            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.size[0] = 7;
            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.size[1] = 7;

            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.center_offset[0] = 0;
            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.center_offset[1] = 0;

            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.stride[0] = 1;
            workflow_item[pool5_7x7_s1]->arguments.forward_pooling.stride[1] = 1;

            workflow_item[pool5_7x7_s1]->output_format[0].format = NN_DATA_FORMAT_3D;
            workflow_item[pool5_7x7_s1]->output_format[0].format_3d = { { 1, 1, 1024 } };
        }

        // Loss3_clasifier
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[pool5_7x7_s1], 0 };
            di->workflow_item_create_function(&workflow_item[loss3_classifier], 1, &inputs_descriptor, 1);

            workflow_item[loss3_classifier]->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
            workflow_item[loss3_classifier]->name = "loss3_classifier";

            workflow_item[loss3_classifier]->arguments.forward_fully_connected.activation.function = NN_ACTIVATION_FUNCTION_NONE;

            workflow_item[loss3_classifier]->arguments.forward_fully_connected.weights = workflow_item_factor[loss3_classifier_weights];
            workflow_item[loss3_classifier]->arguments.forward_fully_connected.biases = workflow_item_factor[loss3_classifier_biases];

            workflow_item[loss3_classifier]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_item[loss3_classifier]->output_format[0].format_1d = { { 1000 } };
        }

        // Softmax
        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[loss3_classifier], 0 };
            di->workflow_item_create_function(&workflow_item[softmax], 1, &inputs_descriptor, 1);

            workflow_item[softmax]->type = NN_WORK_ITEM_TYPE_SOFTMAX;
            workflow_item[softmax]->name = "softmax";

            workflow_item[softmax]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_item[softmax]->output_format[0].format_1d = { { 1000 } };
        }

        {
            nn_workflow_use_descriptor_t inputs_descriptor = { workflow_item[softmax], 0 };
            di->workflow_item_create_function(&workflow_item[output], 1, &inputs_descriptor, 1);

            workflow_item[output]->type = NN_WORK_ITEM_TYPE_OUTPUT;

            workflow_item[output]->output_format[0].format = NN_DATA_FORMAT_1D;
            workflow_item[output]->output_format[0].format_1d = { { 1000 } };

        }

        // -------------------------------------------------------------------------------------------
        // END of workflow stages definition
        // -------------------------------------------------------------------------------------------
        workflow->input[0] = workflow_item[input];
        workflow->output[0] = workflow_item[output];
        // -------------------------------------------------------------------------------------------

        return workflow;
    }
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran before main execution starts.
// The sole function of this construction is attaching this workflow builder to
// library of workflow builders (singleton command pattern).
namespace {
    struct attach {
        workflow_builder_googlenet_float builder;
        attach() {
            workflow_builder::instance().add("googlenet_float", &builder);
        }
    };

    attach attach_;
}
