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

#pragma once

#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "device/api/nn_primitives_api_0.h"

struct nn_workload_item;
struct nn_device_internal;

namespace layer {
class fully_connected_f32 : public nn_primitive_t {
  public:
    fully_connected_f32(size_t num_input,
                        size_t num_output,
                        const nn_argument_activation_t &activation,
                        size_t batch_size,
                        nn_device_internal *device);

    fully_connected_f32(size_t input_size_x,
                        size_t input_size_y,
                        size_t input_size_z,
                        size_t num_output,
                        const nn_argument_activation_t &activation,
                        size_t batch_size,
                        nn_device_internal *device);

    virtual ~fully_connected_f32() {}

    virtual void forward(const std::vector<const nn_workload_data_t *> &inputs,
                         const std::vector<const nn_workload_data_t *> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) override;

    virtual std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta = false) override;

    virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta = false) override;

    virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta = false) override;

    virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

    void backward(const std::vector<nn_workload_data_t *> &inputs,
                  const std::vector<const nn_workload_data_t *> &parameters,
                  const std::vector<const nn_workload_data_t *> &outputs) override;

    void backward_parameter(size_t parameter_index,
                            const std::vector<const nn_workload_data_t *> &inputs,
                            const std::vector<nn_workload_data_t *> &parameters,
                            const std::vector<const nn_workload_data_t *> &outputs) override;

    virtual void dispatch_backward_weights_delta(const nn::workload_data<> *forward_input_view,
                                                 const nn::workload_data<> *backward_input_view,
                                                 nn::workload_data<> *backward_weights_delta_view);

    virtual void dispatch_backward_bias_delta(const nn::workload_data<> *backward_input_view,
                                              const nn::workload_data<> *bias_multiplier,
                                              nn::workload_data<> *backward_bias_delta_view);

    virtual void dispatch_backward_input_delta(const nn::workload_data<> *backward_input_view,
                                               const nn::workload_data<> *forward_weights_view,
                                               nn::workload_data<> *backward_output_view);

    bool get_has_3d_input() {return has_3d_input;}

    uint32_t get_input_size_x() {return input_size_x;}
    uint32_t get_input_size_y() {return input_size_y;}
    uint32_t get_input_size_z() {return input_size_z;}
    uint32_t get_input_size() {return num_input;}
    uint32_t get_output_size() {return num_output;}

  private:
    virtual void forward(const nn::workload_data<> *input,
                         const nn::workload_data<> *weights,
                         const nn::workload_data<> *bias,
                         nn::workload_data<> *output);

    template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
    void run_fully_connected_work_item_internal_batch8(const nn::workload_data<> *input,
                                                       const nn::workload_data<> *weights,
                                                       const nn::workload_data<> *bias,
                                                       nn::workload_data<> *output);
    template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
    void run_fully_connected_work_item_internal_batch48(const nn::workload_data<> *input,
                                                        const nn::workload_data<> *weights,
                                                        const nn::workload_data<> *bias,
                                                        nn::workload_data<> *output);
    template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
    void run_fully_connected_work_item_internal_latency(const nn::workload_data<> *input,
                                                        const nn::workload_data<> *weights,
                                                        const nn::workload_data<> *bias,
                                                        nn::workload_data<> *output);
    template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
    void choose_fully_connected_work_item_batching_mode(const nn::workload_data<> *input,
                                                        const nn::workload_data<> *weights,
                                                        const nn::workload_data<> *bias,
                                                        nn::workload_data<> *output);
    template <bool T_NEED_BIAS_COPY>
    void choose_fully_connected_work_item_activation(const nn::workload_data<> *input,
                                                     const nn::workload_data<> *weights,
                                                     const nn::workload_data<> *bias,
                                                     nn::workload_data<> *output);
    void run_fully_connected_work_item(const nn::workload_data<> *input,
                                       const nn::workload_data<> *weights,
                                       const nn::workload_data<> *bias,
                                       nn::workload_data<> *output);

    friend void unpack_fully_connected_callback_handle(void *void_handle);
    friend void unpack_fully_connected_callback_handle_backward_bias(void *void_handle);
    friend void unpack_fully_connected_callback_handle_backward_weight(void *void_handle);
    friend void unpack_fully_connected_callback_handle_backward_input(void *void_handle);

  protected:
    size_t input_size_x, input_size_y, input_size_z; /* TODO: temporary solution for create_input_delta */
    bool has_3d_input;
    const size_t num_input, num_output, batch_size;
    const nn_argument_activation_t activation;
    nn_device_internal *device;

    static const nn_workload_data_layout_t& in_out_layout;
};

void run_multithreaded_FC_work_item_backward(nn_workload_item *const work_item);
}
