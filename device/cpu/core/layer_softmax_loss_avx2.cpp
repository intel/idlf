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

#include "device/common/nn_workload_data.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_softmax_loss_avx2.h"
#include "device/cpu/api_internal/data_helper.h"

#include <cfloat>
#include <immintrin.h>
#include <string.h>
#include <algorithm>

namespace layer {
///////////////////////////////////////////////////////////////////////////////////////////////////
// forward implementation

void softmax_loss_f32::backward(const std::vector<nn_workload_data_t *> &inputs,
                           const std::vector<const nn_workload_data_t *> &parameters,
                           const std::vector<const nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 2);
    assert(parameters.size() == 0);
    assert(outputs.size() == 1);

    const nn::workload_data<nn::layout_f32> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
    nn::workload_data<nn::layout_f32> backward_output(inputs[0]->parent->delta_buffer, inputs[0]->parent->lengths, inputs[0]->parent->layout);

    backward( &backward_input, reinterpret_cast<const nn::workload_data<int32_t> *>(inputs[1]), &backward_output);
}

void softmax_loss_f32::forward(const nn::workload_data<nn::layout_f32> *input, const nn::workload_data<int32_t> *labels, nn::workload_data<nn::layout_f32> *output, nn::workload_data<nn::layout_f32> *output_loss)
{
    // Run standard forward pass.

    softmax.forward(input, output);

    float loss = 0.0f;

    for(uint32_t n = output->view_begin.t[NN_DATA_COORD_n]; n <= output->view_end.t[NN_DATA_COORD_n]; ++n)
    {
        // This label is provided purely by user - we need runtime security
        // check to ensure it's not used to access data not in buffer.
        // This part is so computationally cheap that we can afford it.
        // We don't need this actually because we use data wrapper - but
        // I left it for future reference during optimizations that will make
        // it work on raw buffers instead of wrappers.
        auto label = static_cast<uint32_t>((*labels)(n, 0, 0, 0, 0, 0));

        if(label < output->parent->lengths.t[NN_DATA_COORD_x])
        {
            loss -= std::log(std::max((*output)(n, label, 0, 0, 0, 0), FLT_MIN));
        }
        else
        {
            throw std::invalid_argument("softmax_loss: label id too big");
        }
    }

    (*output_loss)(0,0,0,0,0,0) = loss / float(output->get_length(NN_DATA_COORD_n));
}

void softmax_loss_f32::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 2);
    assert(parameters.size() == 0);
    assert(outputs.size() == 2);

    forward(reinterpret_cast<const nn::workload_data<nn::layout_f32> *>(inputs[0]),
            reinterpret_cast<const nn::workload_data<int32_t> *>(inputs[1]),
            reinterpret_cast<nn::workload_data<nn::layout_f32> *>(outputs[0]),
            reinterpret_cast<nn::workload_data<nn::layout_f32> *>(outputs[1]));
}

void softmax_loss_f32::backward(
    const nn::workload_data<nn::layout_f32> *input,
    const nn::workload_data<int32_t> *labels,
    nn::workload_data<nn::layout_f32> *output)
{
    const auto C_loss_weight = 1.0f; // Loss weight from caffe.
    const auto C_scale = C_loss_weight / output->get_length(NN_DATA_COORD_n);

    auto output_buffer = static_cast<float*>(output->parent->data_buffer);

    memcpy(output->parent->data_buffer, input->parent->data_buffer, output->parent->buffer_size);

    for(uint32_t n = output->view_begin.t[NN_DATA_COORD_n]; n <= output->view_end.t[NN_DATA_COORD_n]; ++n)
    {
        // This label is provided purely by user - we need runtime security
        // check to ensure it's not used to access data not in buffer.
        // This part is so computationally cheap that we can afford it.
        // We don't need this actually because we use data wrapper - but
        // I left it for future reference during optimizations that will make
        // it work on raw buffers instead of wrappers.
        auto label = static_cast<uint32_t>((*labels)(n, 0, 0, 0, 0, 0));

        if(label < output->parent->lengths.t[NN_DATA_COORD_x])
        {
            (*output)(n, label, 0, 0, 0, 0) -= 1.0f;
        }
        else
        {
            throw std::invalid_argument("softmax_loss: label id too big");
        }

        for (uint32_t x = output->view_begin.t[NN_DATA_COORD_x];
                x <= output->view_end.t[NN_DATA_COORD_x];
                ++x)
        {
            *(output_buffer + x * batch_size + n) *= C_scale;
        }
    }
}

void run_softmax_loss_backward(nn_workload_item *const work_item)
{
    auto primitive = static_cast<softmax_loss_f32*>(work_item->forward_item->primitive);
    primitive->backward(reinterpret_cast<nn::workload_data<nn::layout_f32> *>(work_item->input[0].get_data_view()),
                        reinterpret_cast<nn::workload_data<int32_t> *>(work_item->input[1].get_data_view()),
                        reinterpret_cast<nn::workload_data<nn::layout_f32> *>(work_item->output[0]));
}

softmax_loss_f32::softmax_loss_f32(size_t num_features, size_t batch_size, nn_device_internal *device)
    : num_features(num_features),
      batch_size(batch_size),
      device(device),
      in_out_layout(nn::layout_t<nn::layout_nxyzpq_f32>::layout),
      softmax(num_features, batch_size, device)
{
}

std::vector<nn_workload_data_t *> softmax_loss_f32::create_inputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device,
                                                                     static_cast<uint32_t>(num_features),
                                                                     static_cast<uint32_t>(batch_size),
                                                                     allocate_delta),
            nn::data_helper<NN_WORKLOAD_DATA_TAG_O, int32_t>::create( device, batch_size, false ) };
}

std::vector<nn_workload_data_t *> softmax_loss_f32::create_outputs(bool allocate_delta)
{
    return
    {
        nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device,
                                                                static_cast<uint32_t>(num_features),
                                                                static_cast<uint32_t>(batch_size),
                                                                allocate_delta),
        nn::data_helper<NN_WORKLOAD_DATA_TAG_O, nn::layout_o_f32>::create(device, 1, allocate_delta)
    };
}

bool softmax_loss_f32::validate_input(size_t index, nn_workload_data_t *data)
{
    switch(index){
    case 0:
        return nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::validate(data, static_cast<uint32_t>(num_features), static_cast<uint32_t>(batch_size));
    }

    throw std::invalid_argument("index out of range");
}

} // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_softmax_loss_f32_create_0(
    nn_device_t *device,  /* IDLF device handle */
    size_t num_features,  /* number of input feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_softmax_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);

    std::remove_const<std::remove_pointer<decltype(hints)>::type>::type hints_ = {};
    if (hints != nullptr)
        hints_ = *hints;

    return new layer::softmax_loss_f32(num_features, batch_size, reinterpret_cast<nn_device_internal *>(device));
}
