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

#include "primitives_workload.h"
#include "common/nn_data_tools.h"

#include <map>
#include <fstream>
#include <cmath>
#include <limits>

struct primitives_workload_caffenet_int16_layers
{
    nn_primitive_handle_t a1;
    nn_primitive_handle_t convert_float_int16;
    nn_primitive_handle_t c1;
    nn_primitive_handle_t p1;
    nn_primitive_handle_t n1;

    nn_primitive_handle_t c2g;
    nn_primitive_handle_t p2;
    nn_primitive_handle_t n2;

    nn_primitive_handle_t c3;
    nn_primitive_handle_t c4g;
    nn_primitive_handle_t c5g;
    nn_primitive_handle_t p5;

    nn_primitive_handle_t convert5;

    nn_primitive_handle_t fc6;
    nn_primitive_handle_t fc7;
    nn_primitive_handle_t fc8;

    nn_primitive_handle_t convert8;

    nn_primitive_handle_t sf;

    std::map<std::string, nn_opaque_data_t*> inputs;
    std::map<std::string, nn_opaque_data_t*> outputs;
    std::map<std::string, nn_opaque_data_t*> weights;
    std::map<std::string, nn_opaque_data_t*> bias;
    std::map<std::string, nn_event_t> events;
};

class primitives_workload_caffenet_int16 : public primitives_workload_base {

  private:
    primitives_workload_caffenet_int16_layers layers;

  public:
    primitives_workload_caffenet_int16() : primitives_workload_base(227, false, fi::resize_image_to_square) {
        read_file_to_vector(labels, "weights_caffenet/names.txt", false);
        read_file_to_vector(wwids, "weights_caffenet/wwids.txt", false);
    }

  private:
    nn::data<float> *load_nn_data_from_file(std::string filename) {
        nn::data<float> *data = nn_data_load_from_file_time_measure(filename);
        if (data == nullptr) {
            std::cerr << "Can't load " << filename << std::endl;
            throw;
        }
        return data;
    }

#undef max
#undef min

    template<typename T = int16_t>
    static inline T float2int(float value, float scale) {
        auto scaled = round(value * scale);
        if (scaled > std::numeric_limits<T>::max())
            return std::numeric_limits<T>::max();
        else if (scaled < std::numeric_limits<T>::min())
            return std::numeric_limits<T>::min();
        return scaled;
    }

    template <class T_primitives>
    void load_weights_and_biases(nn_device_t *device,
                                 T_primitives &primitives,
                                 std::string layer_name,
                                 nn_primitive_handle_t &layer_primitive,
                                 std::string weights_file,
                                 std::string biases_file,
                                 std::vector<nn_event_t> &ready_events,
                                 std::vector<nn_data_t *> &temporary_data,
                                 bool i16 = false,
                                 size_t bias_fraction_bits = 16,
                                 size_t weights_fraction_bits = 8) {
        auto weights = load_nn_data_from_file(weights_file);
        auto bias = load_nn_data_from_file(biases_file);
        temporary_data.push_back(weights);
        temporary_data.push_back(bias);

        assert(layer_primitive != NULL);

        nn_opaque_data_t *parameters[2];

        primitives.create_parameters(layer_primitive, 2, parameters, 0, nullptr);
        if (!i16){
            ready_events.push_back(primitives.copy_to_opaque_async(device, parameters[0], weights, 0, nullptr, nullptr));
            ready_events.push_back(primitives.copy_to_opaque_async(device, parameters[1], bias, 0, nullptr, nullptr));
        }
        else{
            auto w16 = new nn::data<int16_t, 0>(static_cast<nn_data_t *>(weights)->size, weights->dimension);
            auto b16 = new nn::data<int32_t, 1>(static_cast<nn_data_t *>(bias)->size, bias->dimension);
            temporary_data.push_back(w16);
            temporary_data.push_back(b16);

            auto w_count = weights->count();
            for (size_t w = 0; w < w_count; ++w)
                ((int16_t *)w16->buffer)[w] = float2int(((float *)weights->buffer)[w], 1 << weights_fraction_bits);

            auto b_count = bias->count();
            for (size_t b = 0; b < b_count; ++b)
                ((int32_t *)b16->buffer)[b] = float2int<int32_t>(((float *)bias->buffer)[b], 1 << bias_fraction_bits);

            ready_events.push_back(primitives.copy_to_opaque_async(device, parameters[0], w16, 0, nullptr, nullptr));
            ready_events.push_back(primitives.copy_to_opaque_async(device, parameters[1], b16, 0, nullptr, nullptr));
        }

        layers.weights[layer_name] = parameters[0];
        layers.bias[layer_name] = parameters[1];
    }

  public:
    virtual bool is_valid() override { return device != nullptr; }

    virtual void init(nn_primitives_0_t primitives_, nn_device_t *device_, size_t batch_size_) override
    {
        primitives = primitives_;
        device = device_;
        batch_size = batch_size_;

        nn_argument_activation activation;
        std::vector<nn_event_t> parameters_ready_events;
        std::vector<nn_data_t *> temporary_data;

        nn_primitives_convolution_hints_t convolution_hints = {};
        nn_primitives_normalization_response_across_maps_hints_t lrn_hints = {};
        nn_primitives_fully_connected_hints_t fc_hints = {};
        nn_primitives_softmax_hints_t sf_hints = {};

        /* Arithmetic layer */
        layers.a1 = primitives.create_handle.arithmetic_f32(
            device, // IDLF device handle
            227,
            227,
            3,
            NN_ARITHMETIC_FUNCTION_SUBTRACTION,
            batch_size, // size of input batch
            nullptr);

        primitives.create_outputs(layers.a1, 1, &layers.outputs["A1"], 0, nullptr);
        primitives.create_parameters(layers.a1, 1, &layers.weights["A1"], 0, nullptr);
        parameters_ready_events.push_back(
            primitives.copy_to_opaque_async(device,
                                            layers.weights["A1"],
                                            load_nn_data_from_file("weights_caffenet/imagenet_mean.nnd"),
                                            0,
                                            nullptr,
                                            nullptr));

        layers.convert_float_int16 = primitives.create_handle.convert_float_to_i16(
            device,
            227,  /* image width */
            227,  /* image height */
            3,    /* number of input feature maps */
            batch_size,
            nullptr,
            nullptr);

        primitives.create_outputs(layers.convert_float_int16, 1, &layers.outputs["Convert_f32_i16"], 0, nullptr);

        convolution_hints.output_padding = {0, 0, 0, 0};
        convolution_hints.fixed_point_fraction_bits.accumulator = 16;
        convolution_hints.fixed_point_fraction_bits.output = 3;

        /* C1 layer */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c1 = primitives.create_handle.convolution_i16(device,
                                                                  11, // kernel_w
                                                                  11, // kernel_h
                                                                  4,  // num_input
                                                                  96, // num_output
                                                                  55, // output width
                                                                  55, // output height
                                                                  0,  // center_offset_x
                                                                  0,  // center_offset_y
                                                                  4,  // stride_x
                                                                  4,  // stride_y
                                                                  &activation,
                                                                  batch_size,
                                                                  &convolution_hints,
                                                                  nullptr);

        // primitives.create_inputs(layers.c1, 1, &layers.inputs["C1"], 0, nullptr);
        primitives.create_outputs(layers.c1, 1, &layers.outputs["C1"], 0, nullptr);
        load_weights_and_biases(device,
                                primitives,
                                "C1",
                                layers.c1,
                                "weights_caffenet/conv1.nnd",
                                "weights_caffenet/conv1_bias.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                16,
                                16);

        /* P1 layer */
        layers.p1 = primitives.create_handle.pooling_i16(
            device,
            NN_POOLING_MODE_MAX, // pooling mode
            3, // pool size x
            3, // pool size y
            2, // pool stride x
            2, // pool stride y
            96, // num of feature maps
            27, // output width
            27, // output height
            batch_size,
            nullptr,
            nullptr);

        primitives.create_outputs(layers.p1, 1, &layers.outputs["P1"], 0, nullptr);

        /* LRN layer */
        lrn_hints = {};
        lrn_hints.fixed_point_fraction_bits.accumulator = 3;
        lrn_hints.fixed_point_fraction_bits.output = 3;
        lrn_hints.output_padding = {2, 2, 2, 2};

        layers.n1 = primitives.create_handle.normalization_response_across_maps_i16(
            device,      // IDLF device handle
            0.0001f / 5, // sum scale
            0.75f,       //  sum power
            1,           // square sum weight
            5,           // size of moving window on the feature maps
            27,          // image width
            27,          // image height
            96,          // number of feature maps
            batch_size,  // size of input batch,
            &lrn_hints,
            nullptr);

        //primitives.create_inputs(layers.n1, 1, &layers.inputs["N1"], 0, nullptr);
        //primitives.validate_input(layers.n1, 0, layers.outputs["P1"], nullptr);
        primitives.create_outputs(layers.n1, 1, &layers.outputs["N1"], 0, nullptr);

        convolution_hints = {};
        convolution_hints.output_padding = { 0, 0, 0, 0 };
        convolution_hints.fixed_point_fraction_bits.accumulator = 17;
        convolution_hints.fixed_point_fraction_bits.output = 7;

        /* C2 layer (each group) */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c2g = primitives.create_handle.convolution_i16(
            device,
            5,       // kernel_w
            5,       // kernel_h
            96 / 2,  // num_input
            256 / 2, // num_output
            27,      // output_w
            27,      // output_h
            2,       // center_offset_x
            2,       // center_offset_y
            1,       // stride_x
            1,       // stride_y
            &activation,
            batch_size,
            &convolution_hints,
            nullptr);

        {
            nn_opaque_data_t *c2_inputs[2];
            primitives.split_z(2, c2_inputs, layers.outputs["N1"]);
            layers.inputs["C2_1"] = c2_inputs[0];
            layers.inputs["C2_2"] = c2_inputs[1];

            nn_opaque_data_t *c2_outputs[2];
            primitives.create_outputs(layers.c2g, 1, &c2_outputs[0], 0, nullptr);
            primitives.create_outputs(layers.c2g, 1, &c2_outputs[1], 0, nullptr);

            primitives.merge_z(&layers.outputs["C2"], 2, c2_outputs);
            layers.outputs["C2_1"] = c2_outputs[0];
            layers.outputs["C2_2"] = c2_outputs[1];
        }

        load_weights_and_biases(device,
            primitives,
            "C2_1",
            layers.c2g,
            "weights_caffenet/conv2_g1.nnd",
            "weights_caffenet/conv2_bias_g1.nnd",
            parameters_ready_events,
            temporary_data,
            true,
            17,
            14);

        load_weights_and_biases(device,
            primitives,
            "C2_2",
            layers.c2g,
            "weights_caffenet/conv2_g2.nnd",
            "weights_caffenet/conv2_bias_g2.nnd",
            parameters_ready_events,
            temporary_data,
            true,
            17,
            14);

        /* P2 layer */
        layers.p2 = primitives.create_handle.pooling_i16(
            device,
            NN_POOLING_MODE_MAX, // pooling mode
            3, // pool size x
            3, // pool size y
            2, // pool stride x
            2, // pool stride y
            256, // num of feature maps
            13, // output width
            13, // output height
            batch_size,
            nullptr,
            nullptr);

        primitives.create_outputs(layers.p2, 1, &layers.outputs["P2"], 0, nullptr);

        /* LRN layer */
        lrn_hints = {};
        lrn_hints.fixed_point_fraction_bits.accumulator = 7;
        lrn_hints.fixed_point_fraction_bits.output = 7;
        lrn_hints.output_padding = {1, 1, 1, 1};

        layers.n2 = primitives.create_handle.normalization_response_across_maps_i16(
            device,      // IDLF device handle
            0.0001f / 5, // sum scale
            0.75f,       //  sum power
            1,           // square sum weight
            5,           // size of moving window on the feature maps
            13,          // image width
            13,          // image height
            256,         // number of feature maps
            batch_size,  // size of input batch
            &lrn_hints,
            nullptr);

        // validate N2 input
        //primitives.create_inputs(layers.n2, 1, &layers.inputs["N2"], 0, nullptr);
        primitives.create_outputs(layers.n2, 1, &layers.outputs["N2"], 0, nullptr);

        /* C3 layer */
        convolution_hints = {};
        convolution_hints.fixed_point_fraction_bits.accumulator = 22;
        convolution_hints.fixed_point_fraction_bits.output = 6;
        convolution_hints.output_padding = {1, 1, 1, 1};

        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c3 = primitives.create_handle.convolution_i16(
            device,
            3, // kernel_w
            3, // kernel_h
            256, // num_input
            384, // num_output
            13, // output_w
            13, // output_h
            1, // center_offset_x
            1, // center_offset_y
            1, // stride_x
            1, // stride_y
            &activation,
            batch_size,
            &convolution_hints,
            nullptr);

        // validate C3 input
        primitives.create_outputs(layers.c3, 1, &layers.outputs["C3"], 0, nullptr);
        load_weights_and_biases(device,
                                primitives,
                                "C3",
                                layers.c3,
                                "weights_caffenet/conv3.nnd",
                                "weights_caffenet/conv3_bias.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                22,
                                15);

        /* C4 layer (each group) */
        convolution_hints = {};
        convolution_hints.fixed_point_fraction_bits.accumulator = 22;
        convolution_hints.fixed_point_fraction_bits.output = 7;
        convolution_hints.output_padding = { 1, 1, 1, 1 };

        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c4g = primitives.create_handle.convolution_i16(
            device,
            3, // kernel_w
            3, // kernel_h
            384 / 2, // num_input
            384 / 2, // num_output
            13, // output_w
            13, // output_h
            1, // center_offset_x
            1, // center_offset_y
            1, // stride_x
            1, // stride_y
            &activation,
            batch_size,
            &convolution_hints,
            nullptr);

        {
            nn_opaque_data_t *c4_inputs[2];
            primitives.split_z(2, c4_inputs, layers.outputs["C3"]);
            layers.inputs["C4_1"] = c4_inputs[0];
            layers.inputs["C4_2"] = c4_inputs[1];
        }

        primitives.create_outputs(layers.c4g, 1, &layers.outputs["C4_1"], 0, nullptr);
        load_weights_and_biases(device,
                                primitives,
                                "C4_1",
                                layers.c4g,
                                "weights_caffenet/conv4_g1.nnd",
                                "weights_caffenet/conv4_bias_g1.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                22,
                                16);

        primitives.create_outputs(layers.c4g, 1, &layers.outputs["C4_2"], 0, nullptr);
        load_weights_and_biases(device,
                                primitives,
                                "C4_2",
                                layers.c4g,
                                "weights_caffenet/conv4_g2.nnd",
                                "weights_caffenet/conv4_bias_g2.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                22,
                                16);

        /* C5 layer (each group) */
        convolution_hints = {};
        convolution_hints.fixed_point_fraction_bits.accumulator = 22;
        convolution_hints.fixed_point_fraction_bits.output = 8;
        convolution_hints.output_padding = { 0, 0, 0, 0 };

        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c5g = primitives.create_handle.convolution_i16(
            device,
            3, // kernel_w
            3, // kernel_h
            384 / 2, // num_input
            256 / 2, // num_output
            13, // output_w
            13, // output_h
            1, // center_offset_x
            1, // center_offset_y
            1, // stride_x
            1, // stride_y
            &activation,
            batch_size,
            &convolution_hints,
            nullptr);

        {
            nn_opaque_data_t *c5_views[2];
            primitives.create_outputs(layers.c5g, 1, &c5_views[0], 0, nullptr);
            primitives.create_outputs(layers.c5g, 1, &c5_views[1], 0, nullptr);
            primitives.merge_z(&layers.outputs["C5"], 2, c5_views);
            layers.outputs["C5_1"] = c5_views[0];
            layers.outputs["C5_2"] = c5_views[1];
        }

        load_weights_and_biases(device,
                                primitives,
                                "C5_1",
                                layers.c5g,
                                "weights_caffenet/conv5_g1.nnd",
                                "weights_caffenet/conv5_bias_g1.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                22,
                                15);

        load_weights_and_biases(device,
                                primitives,
                                "C5_2",
                                layers.c5g,
                                "weights_caffenet/conv5_g2.nnd",
                                "weights_caffenet/conv5_bias_g2.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                22,
                                15);

        /* Pooling P5 */
        layers.p5 = primitives.create_handle.pooling_i16(
            device,
            NN_POOLING_MODE_MAX, // pooling mode
            3, // pool size x
            3, // pool size y
            2, // pool stride x
            2, // pool stride y
            256, // num of feature maps
            6, // output width
            6, // output height
            batch_size,
            nullptr,
            nullptr);

        primitives.create_outputs(layers.p5, 1, &layers.outputs["P5"], 0, nullptr);

        layers.convert5 = primitives.create_handle.convert_z_block_xyz_x2nx_i16(device,
            6, // input width
            6, // input height
            256, // num of input feature maps
            batch_size,
            nullptr);
        primitives.create_outputs(layers.convert5, 1, &layers.inputs["FC6"], 0, nullptr);

        /* FC6 layer */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        fc_hints = {};
        fc_hints.fixed_point_fraction_bits.accumulator = 24;
        fc_hints.fixed_point_fraction_bits.output = 10;
        layers.fc6 = primitives.create_handle.fully_connected_i16(
            device,
            6 * 6 * 256, // num_input
            4096,        // num_output
            &activation,
            batch_size,
            &fc_hints,
            nullptr);

        // validate FC6 input
        //primitives.create_inputs(layers.fc6, 1, &layers.inputs["FC6"], 0, nullptr);
        primitives.create_outputs(layers.fc6, 1, &layers.outputs["FC6"], 0, nullptr);
        load_weights_and_biases(device,
                                primitives,
                                "FC6",
                                layers.fc6,
                                "weights_caffenet/fc6.nnd",
                                "weights_caffenet/fc6_bias.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                24,
                                16);

        /* FC7 layer */
        fc_hints = {};
        fc_hints.fixed_point_fraction_bits.accumulator = 26;
        fc_hints.fixed_point_fraction_bits.output = 12;

        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.fc7 = primitives.create_handle.fully_connected_i16(
            device,
            4096, // num_input
            4096, // num_output
            &activation,
            batch_size,
            &fc_hints,
            nullptr);

        // validate FC7 input
        //assert(primitives.validate_input(layers.fc7, 0, layers.outputs["FC6"]));
        primitives.create_outputs(layers.fc7, 1, &layers.outputs["FC7"], 0, nullptr);
        load_weights_and_biases(device,
                                primitives,
                                "FC7",
                                layers.fc7,
                                "weights_caffenet/fc7.nnd",
                                "weights_caffenet/fc7_bias.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                26,
                                16);

        /* FC8 layer */
        fc_hints = {};
        fc_hints.fixed_point_fraction_bits.accumulator = 24;
        fc_hints.fixed_point_fraction_bits.output = 26;

        activation.function = NN_ACTIVATION_FUNCTION_NONE;
        layers.fc8 = primitives.create_handle.fully_connected_i16_i32(device,
                                                              4096, // num_input
                                                              1000, // num_output
                                                              &activation,
                                                              batch_size,
                                                              &fc_hints,
                                                              nullptr);

        // validate FC8 input
        //assert(primitives.validate_input(layers.fc8, 0, layers.outputs["FC7"]));
        primitives.create_outputs(layers.fc8, 1, &layers.outputs["FC8"], 0, nullptr);
        load_weights_and_biases(device,
                                primitives,
                                "FC8",
                                layers.fc8,
                                "weights_caffenet/fc8.nnd",
                                "weights_caffenet/fc8_bias.nnd",
                                parameters_ready_events,
                                temporary_data,
                                true,
                                24,
                                12);

        layers.convert8 = primitives.create_handle.convert_z2nz_n8xn_i32(device,
            1000, // num of input feature maps
            batch_size,
            nullptr);
        primitives.create_outputs(layers.convert8, 1, &layers.outputs["convert_fc8sf"], 0, nullptr);

        /* Softmax layer */
        sf_hints = {};
        sf_hints.fixed_point_fraction_bits.accumulator = 26;

        layers.sf = primitives.create_handle.softmax_i32(device,
                                                         1000, // number of feature maps
                                                         batch_size,
                                                         &sf_hints,
                                                         nullptr);

        //primitives.create_inputs(layers.sf, 1, &layers.inputs["SF"], 0, nullptr);
        primitives.create_outputs(layers.sf, 1, &layers.outputs["SF"], 0, nullptr);
    }

    template<typename... Ts>
    void data_f32_to_i16(float scale, nn_opaque_data_t* src, nn_opaque_data_t* dst, Ts... sizes){
        nn::data<float, sizeof...(Ts)> src_data(sizes...);
        nn::data<int16_t, sizeof...(Ts)> dst_data(sizes...);
        auto src_copied = primitives.copy_from_opaque_async(device, &src_data, src, 0, nullptr, nullptr);
        primitives.wait(1, &src_copied);
        primitives.delete_event(src_copied);
        {
            const auto i_count = src_data.count();
            auto pi = (float *)src_data.buffer;
            auto po = (int16_t *)dst_data.buffer;
            for (size_t i = 0; i < i_count; ++i) {
                *po++ = float2int(*pi++, scale);
            }
        }

        auto dst_copied = primitives.copy_to_opaque_async(device, dst, &dst_data, 0, nullptr, nullptr);
        primitives.wait(1, &dst_copied);
        primitives.delete_event(dst_copied);
    }

    template<typename... Ts>
    void data_i16_to_f32(float scale, nn_opaque_data_t* src, nn_opaque_data_t* dst, Ts... sizes){
        nn::data<int16_t, sizeof...(Ts)> src_data(sizes...);
        nn::data<float, sizeof...(Ts)> dst_data(sizes...);
        auto src_copied = primitives.copy_from_opaque_async(device, &src_data, src, 0, nullptr, nullptr);
        primitives.wait(1, &src_copied);
        primitives.delete_event(src_copied);
        {
            const auto i_count = src_data.count();
            auto pi = (int16_t *)src_data.buffer;
            auto po = (float *)dst_data.buffer;
            for (size_t i = 0; i < i_count; ++i) {
                *po++ = *pi++ * scale;
            }
        }

        auto dst_copied = primitives.copy_to_opaque_async(device, dst, &dst_data, 0, nullptr, nullptr);
        primitives.wait(1, &dst_copied);
        primitives.delete_event(dst_copied);
    }

    virtual void execute(const nn::data<float, 4> &input, nn::data<float, 2> &output) override
    {
        const size_t batch_size = input.size[3];
        auto input_internal = primitives.map_input(layers.a1, 0, &input, nullptr);

        layers.events["A1"] = primitives.forward_async(layers.a1, 1, &input_internal, 1, &layers.weights["A1"], 1, &layers.outputs["A1"], 0, nullptr, nullptr);
        primitives.wait(1, &layers.events["A1"]);
        layers.events["Convert_f32_i16"] = primitives.forward_async(layers.convert_float_int16, 1, &layers.outputs["A1"], 0, nullptr, 1, &layers.outputs["Convert_f32_i16"], 1, &layers.events["A1"], nullptr);
        nn_opaque_data_t *c1_params[2] = {layers.weights["C1"], layers.bias["C1"]};
        layers.events["C1"] = primitives.forward_async(layers.c1, 1, &layers.outputs["Convert_f32_i16"], 2, c1_params, 1, &layers.outputs["C1"], 1, &layers.events["Convert_f32_i16"], nullptr);
        layers.events["P1"] = primitives.forward_async(layers.p1, 1, &layers.outputs["C1"], 0, nullptr, 1, &layers.outputs["P1"], 1, &layers.events["C1"], nullptr);
        layers.events["N1"] = primitives.forward_async(layers.n1, 1, &layers.outputs["P1"], 0, nullptr, 1, &layers.outputs["N1"], 1, &layers.events["P1"], nullptr);

        nn_opaque_data_t *c2_params[2][2] = { { layers.weights["C2_1"], layers.bias["C2_1"] },
        { layers.weights["C2_2"], layers.bias["C2_2"] } };
        layers.events["C2_1"] = primitives.forward_async(layers.c2g, 1, &layers.inputs["C2_1"], 2, c2_params[0], 1, &layers.outputs["C2_1"], 1, &layers.events["N1"], nullptr);
        layers.events["C2_2"] = primitives.forward_async(layers.c2g, 1, &layers.inputs["C2_2"], 2, c2_params[1], 1, &layers.outputs["C2_2"], 1, &layers.events["N1"], nullptr);

        nn_event_t c2[2];
        c2[0] = layers.events["C2_1"];
        c2[1] = layers.events["C2_2"];
        layers.events["P2"] = primitives.forward_async(layers.p2, 1, &layers.outputs["C2"], 0, nullptr, 1, &layers.outputs["P2"], 2, c2, nullptr);
        layers.events["N2"] = primitives.forward_async(layers.n2, 1, &layers.outputs["P2"], 0, nullptr, 1, &layers.outputs["N2"], 0, nullptr, nullptr);
        nn_opaque_data_t *c3_params[2] = { layers.weights["C3"], layers.bias["C3"] };
        layers.events["C3"] = primitives.forward_async(layers.c3, 1, &layers.outputs["N2"], 2, c3_params, 1, &layers.outputs["C3"], 1, &layers.events["N2"], nullptr);

        nn_opaque_data_t *c4_params[2][2] = { { layers.weights["C4_1"], layers.bias["C4_1"] },
        { layers.weights["C4_2"], layers.bias["C4_2"] } };
        layers.events["C4_1"] = primitives.forward_async(layers.c4g, 1, &layers.inputs["C4_1"], 2, c4_params[0], 1, &layers.outputs["C4_1"], 1, &layers.events["C3"], nullptr);
        layers.events["C4_2"] = primitives.forward_async(layers.c4g, 1, &layers.inputs["C4_2"], 2, c4_params[1], 1, &layers.outputs["C4_2"], 1, &layers.events["C3"], nullptr);

        nn_opaque_data_t *c5_params[2][2] = { { layers.weights["C5_1"], layers.bias["C5_1"] },
        { layers.weights["C5_2"], layers.bias["C5_2"] } };
        layers.events["C5_1"] = primitives.forward_async(layers.c5g, 1, &layers.outputs["C4_1"], 2, c5_params[0], 1, &layers.outputs["C5_1"], 1, &layers.events["C4_1"], nullptr);
        layers.events["C5_2"] = primitives.forward_async(layers.c5g, 1, &layers.outputs["C4_2"], 2, c5_params[1], 1, &layers.outputs["C5_2"], 1, &layers.events["C4_2"], nullptr);

        nn_event_t c5[2];
        c5[0] = layers.events["C5_1"];
        c5[1] = layers.events["C5_2"];
        layers.events["P5"] = primitives.forward_async(layers.p5, 1, &layers.outputs["C5"], 0, nullptr, 1, &layers.outputs["P5"], 2, c5, nullptr);
        layers.events["convert_c5fc6"] = primitives.forward_async(layers.convert5, 1, &layers.outputs["P5"], 0, nullptr, 1, &layers.inputs["FC6"], 1, &layers.events["P5"], nullptr);
        nn_opaque_data_t *fc6_params[2] = { layers.weights["FC6"], layers.bias["FC6"] };
        layers.events["FC6"] = primitives.forward_async(layers.fc6, 1, &layers.inputs["FC6"], 2, fc6_params, 1, &layers.outputs["FC6"], 1, &layers.events["convert_c5fc6"], nullptr);
        nn_opaque_data_t *fc7_params[2] = { layers.weights["FC7"], layers.bias["FC7"] };
        layers.events["FC7"] = primitives.forward_async(layers.fc7, 1, &layers.outputs["FC6"], 2, fc7_params, 1, &layers.outputs["FC7"], 1, &layers.events["FC6"], nullptr);
        nn_opaque_data_t *fc8_params[2] = { layers.weights["FC8"], layers.bias["FC8"] };
        layers.events["FC8"] = primitives.forward_async(layers.fc8, 1, &layers.outputs["FC7"], 2, fc8_params, 1, &layers.outputs["FC8"], 1, &layers.events["FC7"], nullptr);

        if (batch_size > 1) {
            layers.events["convert_fc8sf"] = primitives.forward_async(layers.convert8, 1, &layers.outputs["FC8"], 0, nullptr, 1, &layers.outputs["convert_fc8sf"], 1, &layers.events["FC8"], nullptr);
            layers.events["SF"] = primitives.forward_async(layers.sf, 1, &layers.outputs["convert_fc8sf"], 0, nullptr, 1, &layers.outputs["SF"], 1, &layers.events["convert_fc8sf"], nullptr);
        }
        else {
            layers.events["SF"] = primitives.forward_async(layers.sf, 1, &layers.outputs["FC8"], 0, nullptr, 1, &layers.outputs["SF"], 1, &layers.events["FC8"], nullptr);
        }

        layers.events["output_ready"] =
            primitives.copy_from_opaque_async(device, &output, layers.outputs["SF"], 1, &layers.events["SF"], nullptr);

        primitives.wait(1, &layers.events["output_ready"]);
    }

    void cleanup() override
    {
        /* ****************************************************************************************** */
        /* Cleanup in memory                                                                          */
        /* ****************************************************************************************** */
        std::cout
            << "Cleanup in memory"
            << std::endl
            << "========================================================"
            << std::endl;

        // Delete events
        for (auto& pair : layers.events)
            primitives.delete_event(pair.second);

        // Delete weights
        for (auto& pair : layers.weights)
            primitives.delete_opaque_data(pair.second);

        // Delete biases
        for (auto& pair : layers.bias)
            primitives.delete_opaque_data(pair.second);

        // Delete inputs
        for (auto& pair : layers.inputs)
            primitives.delete_opaque_data(pair.second);

        // Delete outputs
        for (auto& pair : layers.outputs)
            primitives.delete_opaque_data(pair.second);
    }

    virtual ~primitives_workload_caffenet_int16() {}
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran before main execution starts.
// The sole function of this construction is attaching this workload to
// library of workloads (singleton command pattern).
namespace {
    struct attach {
        primitives_workload_caffenet_int16 workload;
        attach() {
            primitives_workload::instance().add("caffenet_int16", &workload);
        }
    };

    attach attach_;
}
