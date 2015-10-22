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
#include "helper_zxyn_f32.h"

namespace layer {

/* Examples of operation performed by arithmetic primitive
   (alpha, beta, gamma are scalars, imput_A and input_B are of workload_data type):

   (1) output = gamma * ( input_A {ARITHMETIC_FUNCTION} input_B )
       Where {ARITHMETIC_FUNCTION} is addition, subtraction, multiplication or division

   (2) output = alpha * input_A {ARITHMETIC_FUNCTION} beta * input_B
        Where {ARITHMETIC_FUNCTION} is addition or subtraction
*/
enum class scalar_op_type {
    NO_SCALAR_OPERATION,
    MUL_BY_GAMMA,
    MUL_BY_ALPHA,
    MUL_BY_BETA,
    MUL_BY_ALPHA_AND_BETA
};

class arithmetic_f32 : public helper_zxyn_f32::primitive_zxyn_f32_base {
  public:
    arithmetic_f32(size_t image_size_x,
                   size_t image_size_y,
                   size_t image_size_z,
                   NN_ARITHMETIC_FUNCTION arithmetic_function,
                   size_t batch_size,
                   nn_device_internal *device);

    arithmetic_f32(size_t image_size_x,
                   size_t image_size_y,
                   size_t image_size_z,
                   NN_ARITHMETIC_FUNCTION arithmetic_function,
                   size_t batch_size,
                   float alpha,
                   float beta,
                   float gamma,
                   nn_device_internal *device);

    virtual std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta = false) override;

    bool validate_input(size_t index, nn_workload_data_t *data) override;

    void forward(const std::vector<const nn_workload_data_t *> &inputs,
                 const std::vector<const nn_workload_data_t *> &parameters,
                 const std::vector<nn_workload_data_t *> &outputs) override;

    void prepare_forward(const std::vector<const nn_workload_data_t *> &inputs,
                         const std::vector<const nn_workload_data_t *> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) override;

    std::vector<float> get_input_feat_periodic(const std::vector<const nn_workload_data_t *> &parameters) const;
    bool is_linear() const { return (arithmetic_function == NN_ARITHMETIC_FUNCTION_SUBTRACTION)
                                    or (arithmetic_function == NN_ARITHMETIC_FUNCTION_ADDITION); }

  protected:
    const NN_ARITHMETIC_FUNCTION arithmetic_function;
    float alpha, beta, gamma;
    std::vector<nn_multithreaded_request> job;
    std::tuple<float*, float*> prepared_for;

    virtual size_t get_required_input_w() override;
    virtual size_t get_required_input_h() override;

  private:
    void forward(const nn::workload_data<nn::layout_f32> *input,
                 const nn::workload_data<nn::layout_f32> *factor,
                 nn::workload_data<nn::layout_f32> *output);

    void prepare_forward(const nn::workload_data<nn::layout_f32> *input,
                         const nn::workload_data<nn::layout_f32> *factor,
                         nn::workload_data<nn::layout_f32> *output);

    void run_arithmetic_operation_work_item(const nn::workload_data<nn::layout_f32> *input,
                                            const nn::workload_data<nn::layout_f32> *factor,
                                            nn::workload_data<nn::layout_f32> *output);
    template <NN_ARITHMETIC_FUNCTION T_function, scalar_op_type scalar_op>
    void process_arithmetic_operation(const nn::workload_data<nn::layout_f32> *input,
                                      const nn::workload_data<nn::layout_f32> *factor,
                                      nn::workload_data<nn::layout_f32> *output);
    friend void unpack_arithmetic_callback_handle(void *void_handle);
};
} // namespace layer
