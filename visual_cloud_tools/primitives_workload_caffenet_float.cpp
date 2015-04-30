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
#include "nn_data_tools.h"

#include <map>
#include <fstream>

struct primitives_workload_caffenet_float_layers
{
    nn_primitive_handle_t a1;
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
    nn_primitive_handle_t sf;

    std::map<std::string, nn_opaque_data_t*> inputs;
    std::map<std::string, nn_opaque_data_t*> outputs;
    std::map<std::string, nn_opaque_data_t*> weights;
    std::map<std::string, nn_opaque_data_t*> bias;
    std::map<std::string, nn_event_t> events;
};

class primitives_workload_caffenet_float : public primitives_workload_base {

  private:
    primitives_workload_caffenet_float_layers layers;

  public:
    primitives_workload_caffenet_float() : primitives_workload_base(227, false, fi::resize_image_to_square) {
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

    template <class T_primitive>
    void load_weights_and_biases(std::string layer_name,
                                 nn_primitive_handle_t &layer_primitive,
                                 const T_primitive &primitive_methods,
                                 std::string weights_file,
                                 std::string biases_file) {
        auto weights = load_nn_data_from_file(weights_file);
        auto bias = load_nn_data_from_file(biases_file);

        assert(layer_primitive != NULL);

        nn_opaque_data_t *layer_weights = primitive_methods->create_weights(layer_primitive, weights, nullptr);
        nn_opaque_data_t *layer_biases = primitive_methods->create_bias(layer_primitive, bias, nullptr);

        layers.weights[layer_name] = layer_weights;
        layers.bias[layer_name] = layer_biases;

        delete weights;
        delete bias;
    }

  public:
    virtual bool is_valid() override { return device != nullptr; }

    virtual void init(nn_primitives_0_t primitives_, nn_device_t *device_, size_t batch_size_) override
    {
        primitives = primitives_;
        device = device_;
        batch_size = batch_size_;

        nn_argument_activation activation;

        /* Arithmetic layer */
        layers.a1 = primitives.arithmetic_f32->create_handle(
            device, // IDLF device handle
            227,
            227,
            3,
            NN_ARITHMETIC_FUNCTION_SUBTRACTION,
            batch_size, // size of input batch
            nullptr);

        layers.outputs["A1"] = primitives.arithmetic_f32->create_output(layers.a1, nullptr);
        layers.weights["A1"] = primitives.arithmetic_f32->create_factor(layers.a1, load_nn_data_from_file("weights_caffenet/imagenet_mean.nnd"), nullptr);

        /* C1 layer */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c1 = primitives.convolution_f32->create_handle(
            device,
            11, // kernel_w
            11, // kernel_h
            3, // num_input
            96, // num_output
            55, // output_w
            55, // output_h
            0, // center_offset_x
            0, // center_offset_y
            4, // stride_x
            4, // stride_y
            &activation,
            batch_size,
            nullptr);

        // validate C1 input
        //assert(primitives.convolution_f32->validate_input(layers.c1, layers.outputs["A1"]));
        layers.outputs["C1"] = primitives.convolution_f32->create_output_with_padding(layers.c1, 0, 0, 0, 0, nullptr);
        load_weights_and_biases("C1",
                                layers.c1,
                                primitives.convolution_f32,
                                "weights_caffenet/conv1.nnd",
                                "weights_caffenet/conv1_bias.nnd");

        /* P1 layer */
        layers.p1 = primitives.pooling_f32->create_handle(
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
            nullptr);

        // validate P1 input
        assert(primitives.pooling_f32->validate_input(layers.p1, layers.outputs["C1"]));
        layers.outputs["P1"] = primitives.pooling_f32->create_output_with_padding(layers.p1, 0, 0, 0, 0, nullptr);

        /* LRN layer */
        layers.n1 = primitives.normalization_response_across_maps_f32->create_handle(
            device, // IDLF device handle
            0.0001f/5, // sum scale
            0.75f, //  sum power
            1, // square sum weight
            5, // size of moving window on the feature maps
            27, // image width
            27, // image height
            96, // number of feature maps
            batch_size, // size of input batch
            nullptr);

        // validate N1 input
        assert(primitives.normalization_response_across_maps_f32->validate_input(layers.n1, layers.outputs["P1"]));
        layers.outputs["N1"] = primitives.normalization_response_across_maps_f32->create_output_with_padding(layers.n1, 2, 2, 2, 2, nullptr);

        /* C2 layer (each group) */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c2g = primitives.convolution_f32->create_handle(
            device,
            5, // kernel_w
            5, // kernel_h
            96 / 2, // num_input
            256 / 2, // num_output
            27, // output_w
            27, // output_h
            2, // center_offset_x
            2, // center_offset_y
            1, // stride_x
            1, // stride_y
            &activation,
            batch_size,
            nullptr);

        {
            nn_opaque_data_t *c2_inputs[2];
            primitives.convolution_f32->split_input_z(layers.c2g, 2, c2_inputs, layers.outputs["N1"]);
            layers.inputs["C2_1"] = c2_inputs[0];
            layers.inputs["C2_2"] = c2_inputs[1];

            nn_opaque_data_t *c2_outputs[2];
            layers.outputs["C2"] = primitives.convolution_f32->create_output_vector_z(layers.c2g, 2, c2_outputs, nullptr);
            layers.outputs["C2_1"] = c2_outputs[0];
            layers.outputs["C2_2"] = c2_outputs[1];
        }

        //assert(primitives.convolution_f32->validate_input(layers.c2_1, layers.inputs["C2_1"]));
        //assert(primitives.convolution_f32->validate_input(layers.c2_2, layers.inputs["C2_1"]));

        load_weights_and_biases("C2_1",
                                layers.c2g,
                                primitives.convolution_f32,
                                "weights_caffenet/conv2_g1.nnd",
                                "weights_caffenet/conv2_bias_g1.nnd");

        load_weights_and_biases("C2_2",
                                layers.c2g,
                                primitives.convolution_f32,
                                "weights_caffenet/conv2_g2.nnd",
                                "weights_caffenet/conv2_bias_g2.nnd");

        /* P2 layer */
        layers.p2 = primitives.pooling_f32->create_handle(
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
            nullptr);

        // validate P2 input
        assert(primitives.pooling_f32->validate_input(layers.p2, layers.outputs["C2"]));
        layers.outputs["P2"] = primitives.pooling_f32->create_output_with_padding(layers.p2, 0, 0, 0, 0, nullptr);

        /* LRN layer */
        layers.n2 = primitives.normalization_response_across_maps_f32->create_handle(
            device, // IDLF device handle
            0.0001f / 5, // sum scale
            0.75f, //  sum power
            1, // square sum weight
            5, // size of moving window on the feature maps
            13, // image width
            13, // image height
            256, // number of feature maps
            batch_size, // size of input batch
            nullptr);

        // validate N2 input
        assert(primitives.normalization_response_across_maps_f32->validate_input(layers.n2, layers.outputs["P2"]));
        layers.outputs["N2"] = primitives.normalization_response_across_maps_f32->create_output_with_padding(layers.n2, 1, 1, 1, 1, nullptr);

        /* C3 layer */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c3 = primitives.convolution_f32->create_handle(
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
            nullptr);

        // validate C3 input
        //assert(primitives.convolution_f32->validate_input(layers.c3, layers.outputs["N2"]));
        layers.outputs["C3"] = primitives.convolution_f32->create_output_with_padding(layers.c3, 1, 1, 1, 1, nullptr);
        load_weights_and_biases("C3",
                                layers.c3,
                                primitives.convolution_f32,
                                "weights_caffenet/conv3.nnd",
                                "weights_caffenet/conv3_bias.nnd");

        /* C4 layer (each group) */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c4g = primitives.convolution_f32->create_handle(
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
            nullptr);

        {
            nn_opaque_data_t *c4_inputs[2];
            primitives.convolution_f32->split_input_z(layers.c4g, 2, c4_inputs, layers.outputs["C3"]);
            layers.inputs["C4_1"] = c4_inputs[0];
            layers.inputs["C4_2"] = c4_inputs[1];
        }

        //assert(primitives.convolution_f32->validate_input(layers.c4_1, layers.inputs["C4_1"]));
        //assert(primitives.convolution_f32->validate_input(layers.c4_2, layers.inputs["C4_2"]));

        layers.outputs["C4_1"] = primitives.convolution_f32->create_output_with_padding(layers.c4g, 1, 1, 1, 1, nullptr);
        load_weights_and_biases("C4_1",
                                layers.c4g,
                                primitives.convolution_f32,
                                "weights_caffenet/conv4_g1.nnd",
                                "weights_caffenet/conv4_bias_g1.nnd");

        layers.outputs["C4_2"] = primitives.convolution_f32->create_output_with_padding(layers.c4g, 1, 1, 1, 1, nullptr);
        load_weights_and_biases("C4_2",
                                layers.c4g,
                                primitives.convolution_f32,
                                "weights_caffenet/conv4_g2.nnd",
                                "weights_caffenet/conv4_bias_g2.nnd");

        /* C5 layer (each group) */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.c5g = primitives.convolution_f32->create_handle(
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
            nullptr);

        {
            nn_opaque_data_t *c5_views[2];
            layers.outputs["C5"] = primitives.convolution_f32->create_output_vector_z(layers.c5g, 2, c5_views, nullptr);
            layers.outputs["C5_1"] = c5_views[0];
            layers.outputs["C5_2"] = c5_views[1];
        }

        //assert(primitives.convolution_f32->validate_input(layers.c5_1, layers.outputs["C4_1"]));
        //assert(primitives.convolution_f32->validate_input(layers.c5_2, layers.outputs["C4_2"]));

        load_weights_and_biases("C5_1",
                                layers.c5g,
                                primitives.convolution_f32,
                                "weights_caffenet/conv5_g1.nnd",
                                "weights_caffenet/conv5_bias_g1.nnd");

        load_weights_and_biases("C5_2",
                                layers.c5g,
                                primitives.convolution_f32,
                                "weights_caffenet/conv5_g2.nnd",
                                "weights_caffenet/conv5_bias_g2.nnd");

        /* Pooling P5 */
        layers.p5 = primitives.pooling_f32->create_handle(
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
            nullptr);

        layers.outputs["P5"] = primitives.pooling_f32->create_output_with_padding(layers.p5, 0, 0, 0, 0, nullptr);

        layers.convert5 = primitives.convert_zxyn_nx_f32->create_handle(device, 6, 6, 256, batch_size, nullptr);
        layers.inputs["FC6"] = primitives.convert_zxyn_nx_f32->create_output(layers.convert5, nullptr);

        /* FC6 layer */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.fc6 = primitives.fully_connected_f32->create_handle(
            device,
            6*6*256, // num_input
            4096, // num_output
            &activation,
            batch_size,
            nullptr);

        // validate FC6 input
        assert(primitives.fully_connected_f32->validate_input(layers.fc6, layers.inputs["FC6"]));
        layers.outputs["FC6"] = primitives.fully_connected_f32->create_output(layers.fc6, nullptr);
        load_weights_and_biases("FC6",
                                layers.fc6,
                                primitives.fully_connected_f32,
                                "weights_caffenet/fc6.nnd",
                                "weights_caffenet/fc6_bias.nnd");

        /* FC7 layer */
        activation.function = NN_ACTIVATION_FUNCTION_RELU;
        layers.fc7 = primitives.fully_connected_f32->create_handle(
            device,
            4096, // num_input
            4096, // num_output
            &activation,
            batch_size,
            nullptr);

        // validate FC7 input
        assert(primitives.fully_connected_f32->validate_input(layers.fc7, layers.outputs["FC6"]));
        layers.outputs["FC7"] = primitives.fully_connected_f32->create_output(layers.fc7, nullptr);
        load_weights_and_biases("FC7",
                                layers.fc7,
                                primitives.fully_connected_f32,
                                "weights_caffenet/fc7.nnd",
                                "weights_caffenet/fc7_bias.nnd");

        /* FC8 layer */
        activation.function = NN_ACTIVATION_FUNCTION_NONE;
        layers.fc8 = primitives.fully_connected_f32->create_handle(
            device,
            4096, // num_input
            1000, // num_output
            &activation,
            batch_size,
            nullptr);

        // validate FC8 input
        assert(primitives.fully_connected_f32->validate_input(layers.fc8, layers.outputs["FC7"]));
        layers.outputs["FC8"] = primitives.fully_connected_f32->create_output(layers.fc8, nullptr);
        load_weights_and_biases("FC8",
                                layers.fc8,
                                primitives.fully_connected_f32,
                                "weights_caffenet/fc8.nnd",
                                "weights_caffenet/fc8_bias.nnd");

        /* Softmax layer */
        layers.sf = primitives.softmax_f32->create_handle(
            device,
            1000, // number of feature maps
            batch_size,
            nullptr);

        // validate SF input
        assert(primitives.softmax_f32->validate_input(layers.sf, layers.outputs["FC8"]));
        layers.outputs["SF"] = primitives.softmax_f32->create_output(layers.sf, nullptr);
    }

    virtual void execute(const nn::data<float, 4> &input, nn::data<float, 2> &output) override
    {
        const size_t batch_size = input.size[3];
        auto input_internal = primitives.arithmetic_f32->map_input(layers.a1, &input, nullptr);

        layers.events["A1"] = primitives.arithmetic_f32->forward_with_parameters_async(layers.a1, input_internal, layers.weights["A1"], layers.outputs["A1"], 0, nullptr, nullptr);
        layers.events["C1"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c1, layers.outputs["A1"], layers.weights["C1"], layers.bias["C1"], layers.outputs["C1"], 1, &layers.events["A1"], nullptr);
        layers.events["P1"] = primitives.pooling_f32->forward_async(layers.p1, layers.outputs["C1"], layers.outputs["P1"], 1, &layers.events["C1"], nullptr);
        layers.events["N1"] = primitives.normalization_response_across_maps_f32->forward_async(layers.n1, layers.outputs["P1"], layers.outputs["N1"], 1, &layers.events["P1"], nullptr);
        layers.events["C2_1"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c2g, layers.inputs["C2_1"], layers.weights["C2_1"], layers.bias["C2_1"], layers.outputs["C2_1"], 1, &layers.events["N1"], nullptr);
        layers.events["C2_2"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c2g, layers.inputs["C2_2"], layers.weights["C2_2"], layers.bias["C2_2"], layers.outputs["C2_2"], 1, &layers.events["N1"], nullptr);

        nn_event_t c2[2];
        c2[0] = layers.events["C2_1"];
        c2[1] = layers.events["C2_2"];
        layers.events["P2"] = primitives.pooling_f32->forward_async(layers.p2, layers.outputs["C2"], layers.outputs["P2"], 2, c2, nullptr);
        layers.events["N2"] = primitives.normalization_response_across_maps_f32->forward_async(layers.n2, layers.outputs["P2"], layers.outputs["N2"], 0, nullptr, nullptr);
        layers.events["C3"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c3, layers.outputs["N2"], layers.weights["C3"], layers.bias["C3"], layers.outputs["C3"], 1, &layers.events["N2"], nullptr);
        layers.events["C4_1"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c4g, layers.inputs["C4_1"], layers.weights["C4_1"], layers.bias["C4_1"], layers.outputs["C4_1"], 1, &layers.events["C3"], nullptr);
        layers.events["C4_2"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c4g, layers.inputs["C4_2"], layers.weights["C4_2"], layers.bias["C4_2"], layers.outputs["C4_2"], 1, &layers.events["C3"], nullptr);
        layers.events["C5_1"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c5g, layers.outputs["C4_1"], layers.weights["C5_1"], layers.bias["C5_1"], layers.outputs["C5_1"], 1, &layers.events["C4_1"], nullptr);
        layers.events["C5_2"] = primitives.convolution_f32->forward_with_weights_and_bias_async(layers.c5g, layers.outputs["C4_2"], layers.weights["C5_2"], layers.bias["C5_2"], layers.outputs["C5_2"], 1, &layers.events["C4_2"], nullptr);

        nn_event_t c5[2];
        c5[0] = layers.events["C5_1"];
        c5[1] = layers.events["C5_2"];
        layers.events["P5"] = primitives.pooling_f32->forward_async(layers.p5, layers.outputs["C5"], layers.outputs["P5"], 2, c5, nullptr);
        layers.events["convert_c5fc6"] = primitives.convert_zxyn_nx_f32->forward_async(layers.convert5, layers.outputs["P5"], layers.inputs["FC6"], 1, &layers.events["P5"], nullptr);
        layers.events["FC6"] = primitives.fully_connected_f32->forward_with_weights_and_bias_async(layers.fc6, layers.inputs["FC6"], layers.weights["FC6"], layers.bias["FC6"], layers.outputs["FC6"], 1, &layers.events["convert_c5fc6"], nullptr);
        layers.events["FC7"] = primitives.fully_connected_f32->forward_with_weights_and_bias_async(layers.fc7, layers.outputs["FC6"], layers.weights["FC7"], layers.bias["FC7"], layers.outputs["FC7"], 1, &layers.events["FC6"], nullptr);
        layers.events["FC8"] = primitives.fully_connected_f32->forward_with_weights_and_bias_async(layers.fc8, layers.outputs["FC7"], layers.weights["FC8"], layers.bias["FC8"], layers.outputs["FC8"], 1, &layers.events["FC7"], nullptr);
        layers.events["SF"] = primitives.softmax_f32->forward_async(layers.sf, layers.outputs["FC8"], layers.outputs["SF"], 1, &layers.events["FC8"], nullptr);
        layers.events["output_ready"] = primitives.softmax_f32->copy_output_async(layers.sf, &output, layers.outputs["SF"], 1, &layers.events["SF"], nullptr);

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

    virtual ~primitives_workload_caffenet_float() {}
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran before main execution starts.
// The sole function of this construction is attaching this workload to
// library of workloads (singleton command pattern).
namespace {
    struct attach {
        primitives_workload_caffenet_float workload;
        attach() {
            primitives_workload::instance().add("caffenet_float", &workload);
        }
    };

    attach attach_;
}
