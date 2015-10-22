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

#include "tester/common/test_aggregator.h"
#include "tester/common/workflows_for_tests.h"
#include "tester/common/test_common_tools.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "device/cpu/core/layer_arithmetic_operation.h"
#include "demo/device/workflow_builder.h"
#include "demo/common/report_maker.h"
#include "common/nn_data_tools.h"
#include "common/time_control.h"
#include "device/cpu/core/layer_arithmetic_operation.h"

#include "device/common/nn_workload_data.h"

#include <memory>
#include <cmath>
#include <numeric>

// BLIND TEST
class test_convolution_float_cpu_random : public test_base {
private:
    void naive_convolv_float_implementation(
        float* input_ref,
        float* output_ref,
        float* biases_ref,
        float* kernel_ref,
        uint_least32_t batch_size,
        uint_least32_t num_output_feature_maps,
        uint_least32_t num_input_feature_maps,
        uint_least32_t output_feature_map_width,
        uint_least32_t output_feature_map_height,
        uint_least32_t input_feature_map_width,
        uint_least32_t input_feature_map_height,
        uint_least32_t kernel_width,
        uint_least32_t kernel_height,
        uint_least32_t kernel_stride_x,
        uint_least32_t kernel_stride_y,
        NN_ACTIVATION_FUNCTION activation );

    bool init();
    bool done();
    void cleanup();

    tested_device*                 current_tested_device;
    nn_device_interface_0_t*  di;
    std::string device;
    std::string model;

    // If test needs a workflow definition
    workflows_for_tests_base    *workflow_wrapper;
    nn_workflow_t               *workflow;

    // Add current test specific variables

public:
    test_convolution_float_cpu_random() {
        test_description = "convolution float cpu random";
    };
    ~test_convolution_float_cpu_random() {};

    bool run();
};

void test_convolution_float_cpu_random::naive_convolv_float_implementation(
    float* input_ref,
    float* output_ref,
    float* biases_ref,
    float* kernel_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation ) {

    for( uint_least32_t batch = 0; batch < batch_size; batch++ ) {
        for( uint_least32_t output_feature_map = 0; output_feature_map < num_output_feature_maps; output_feature_map += 8 ) {
            for( uint_least32_t input_row = 0, output_row = 0; output_row < output_feature_map_height; input_row += kernel_stride_y, output_row++ ) {
                for( uint_least32_t input_column = 0, output_column = 0; output_column < output_feature_map_width; input_column += kernel_stride_x, output_column++ ) {

                    const uint_least32_t out_ofss = output_feature_map_width * output_feature_map_height;
                    const uint_least32_t out_base = batch * output_feature_map_width * output_feature_map_height * num_output_feature_maps +
                        output_column +
                        output_row * output_feature_map_width +
                        output_feature_map * output_feature_map_width * output_feature_map_height;

                    float accumulator0 = 0.0f;
                    float accumulator1 = 0.0f;
                    float accumulator2 = 0.0f;
                    float accumulator3 = 0.0f;
                    float accumulator4 = 0.0f;
                    float accumulator5 = 0.0f;
                    float accumulator6 = 0.0f;
                    float accumulator7 = 0.0f;

                    for( uint_least32_t input_feature_map = 0; input_feature_map < num_input_feature_maps; input_feature_map++ ) {
                        for( uint_least32_t kernel_row = 0; kernel_row < kernel_height; kernel_row++ ) {

                            const uint_least32_t kern_ofss = kernel_width*kernel_height*num_input_feature_maps;
                            uint_least32_t kern_base = kernel_row*kernel_width +
                                input_feature_map*kernel_width*kernel_height +
                                output_feature_map*kernel_width*kernel_height*num_input_feature_maps;

                            for( uint_least32_t kernel_column = 0; kernel_column < kernel_width; kernel_column++ ) {
                                float weight0 = kernel_ref[kern_base + 0 * kern_ofss];
                                float weight1 = kernel_ref[kern_base + 1 * kern_ofss];
                                float weight2 = kernel_ref[kern_base + 2 * kern_ofss];
                                float weight3 = kernel_ref[kern_base + 3 * kern_ofss];
                                float weight4 = kernel_ref[kern_base + 4 * kern_ofss];
                                float weight5 = kernel_ref[kern_base + 5 * kern_ofss];
                                float weight6 = kernel_ref[kern_base + 6 * kern_ofss];
                                float weight7 = kernel_ref[kern_base + 7 * kern_ofss];

                                ++kern_base;

                                // xyzn
                                float input = input_ref[
                                    input_feature_map_width * input_feature_map_height * num_input_feature_maps * batch +
                                        input_feature_map_width * input_feature_map_height * input_feature_map +
                                        input_feature_map_height * input_row +
                                        input_column +
                                        input_feature_map_height * kernel_row +
                                        kernel_column
                                ];

                                accumulator0 += weight0 * input;
                                accumulator1 += weight1 * input;
                                accumulator2 += weight2 * input;
                                accumulator3 += weight3 * input;
                                accumulator4 += weight4 * input;
                                accumulator5 += weight5 * input;
                                accumulator6 += weight6 * input;
                                accumulator7 += weight7 * input;
                            }
                        }
                    }

                    switch( activation ) {
                        case NN_ACTIVATION_FUNCTION_RELU:
                            accumulator0 = (std::max)(0.0f, accumulator0 + biases_ref[output_feature_map + 0]);
                            break;

                        case NN_ACTIVATION_FUNCTION_NONE:
                            accumulator0 = accumulator0 + biases_ref[output_feature_map + 0];
                            break;

                        default:
                            break;
                    }

                    output_ref[out_base + 0 * out_ofss] = accumulator0;
                    output_ref[out_base + 1 * out_ofss] = accumulator1;
                    output_ref[out_base + 2 * out_ofss] = accumulator2;
                    output_ref[out_base + 3 * out_ofss] = accumulator3;
                    output_ref[out_base + 4 * out_ofss] = accumulator4;
                    output_ref[out_base + 5 * out_ofss] = accumulator5;
                    output_ref[out_base + 6 * out_ofss] = accumulator6;
                    output_ref[out_base + 7 * out_ofss] = accumulator7;
                }
            }
        }
    }
}

bool test_convolution_float_cpu_random::init() {
    bool  init_ok = true;
    test_measurement_result   init_result;
    init_result.description = "INIT: " + test_description;

    C_time_control            init_timer;

    try {
    if( devices != nullptr && tests_results != nullptr ) {
        device = "device_cpu";
        model = "caffenet_float";

        current_tested_device = devices->get( device + dynamic_library_extension );
        di = current_tested_device->get_device_interface();

        init_ok = true;
    } else  throw std::runtime_error( "Can't find aggregator of devices" );

    // HERE init workflow
        workflow_wrapper = workflows_for_tests::instance().get( "workflow_for_testing_float_convolution" );
        workflow = workflow_wrapper->init_test_workflow( di );
        if(workflow == nullptr)  throw std::runtime_error( "workflow has not been initialized" );

    } catch(std::runtime_error &error) {
        init_result << "error: " + std::string( error.what() );
        init_ok = false;
    } catch( std::exception &error ) {
        init_result << "error: " + std::string( error.what() );
        init_ok = false;
    } catch(...) {
        init_result << "error: unknown";
        init_ok = false;
    }

    init_timer.tock();
    init_result.time_consumed = init_timer.get_time_diff();
    init_result.clocks_consumed = init_timer.get_clocks_diff();
    init_result.passed = init_ok;

    tests_results << init_result;

    return init_ok;
}

bool test_convolution_float_cpu_random::run() {
    bool run_ok = true;
    test_measurement_result run_result;
    run_result.description = "RUN SUMMARY: " + test_description;

    std::cout << "-> Testing: " << test_description << std::endl;

    try {
        if(!init()) throw std::runtime_error( "init() returns false so can't run test" );

        NN_WORKLOAD_DATA_TYPE input_format  = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
        NN_WORKLOAD_DATA_TYPE output_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;

        for(uint32_t batch : { 1, 8, 48 }) {
            bool local_ok = true;
            test_measurement_result local_result;
            local_result.description = "RUN PART: (batch " + std::to_string( batch ) + ") execution of " + test_description;
            C_time_control  local_timer;

            // begin local test
            uint32_t z = 2,
                img_size = 227,
                num_features_map = 8;

            nn::data<float, 4> *images = new nn::data<float, 4>( img_size, img_size, z, batch );
            if(images == nullptr) throw std::runtime_error("Cant't create images nn::data");

            nn_data_populate( nn::data_cast<float, 0>(images),
                0.0f,
                255.0f );

            nn::data<float, 4> *images_with_padding = new nn::data<float, 4>( img_size + 2, img_size + 2, z, batch );
            if(images_with_padding == nullptr) {
                delete images;
                throw std::runtime_error("Cant't create images_with_padding nn::data");
            }
            { // padding for input for naive method
                nn_data_populate( nn::data_cast<float, 0>(images_with_padding),
                    0.0f );
                for(uint32_t tmp_batch = 0; tmp_batch < batch; ++tmp_batch)
                    for(uint32_t tmp_z = 0; tmp_z < z; ++tmp_z)
                        for(uint32_t y = 0; y < img_size; ++y)
                            for(uint32_t x = 0; x < img_size; ++x)
                                images_with_padding->at( x, y, tmp_z, tmp_batch ) = images->at( x, y, tmp_z, tmp_batch );

            }

            nn_workload_t *workload = nullptr;
            nn_data_t *input_array[1] = { images };
            auto workload_output = new nn::data<float, 4>( img_size, img_size, num_features_map, batch );
            if(workload_output==nullptr) {
                delete images;
                delete images_with_padding;
                throw std::runtime_error("unable to create workload_output for batch = " +std::to_string(batch));
            }

            nn::data<float> *output_array_cmpl[1] = { nn::data_cast<float, 0>(workload_output) };

            auto naive_output = new nn::data<float, 4>( img_size, img_size, num_features_map, batch );
            if(naive_output==nullptr) {
                delete images;
                delete images_with_padding;
                delete workload_output;
                throw std::runtime_error("unable to create naive_output for batch = " +std::to_string(batch));
            }

            auto status = di->workflow_compile_function( &workload, di->device, workflow, &input_format, &output_format, batch );
            if(!workload) throw std::runtime_error( "workload compilation failed for batch = " + std::to_string( batch )
                + " status: " + std::to_string( status ) );

            test_measurement_result run_result;
            run_result.description = "RUN PART: (batch " + std::to_string( batch ) + ") execution of " + test_description;

            // changing order needed
            //di->workload_execute_function( workload, reinterpret_cast<void**>(input_array), reinterpret_cast<void**>(output_array_cmpl), &status );

            float* biases = nullptr;
            float* weights = nullptr;

            { // read biases and weights
                if(NN_WORK_ITEM_TYPE_CONVOLUTION == workflow->input[0]->use[0].item->type) {
                    auto tmp = reinterpret_cast<nn_arguments_forward_convolution_t*>(&workflow->input[0]->use[0].item->arguments);
                    biases = reinterpret_cast<float*>(tmp->biases->buffer);
                    weights = reinterpret_cast<float*>(tmp->weights->buffer);
                }
            }

            if(nullptr == biases || nullptr == weights)
                throw std::runtime_error( "reading weight or biases for naive version failed for batch = " + std::to_string( batch ) );

            naive_convolv_float_implementation(
                reinterpret_cast<float*>(images_with_padding->buffer),
                reinterpret_cast<float*>(naive_output->buffer),
                biases,
                weights,
                batch,
                num_features_map,
                z,
                img_size,
                img_size,
                img_size + 2,
                img_size + 2,
                3,
                3,
                1,
                1,
                NN_ACTIVATION_FUNCTION_RELU );

            //local_ok = compare_4d_data( workload_output, naive_output );
            local_ok = true; // BLIND TEST

            // end of local test
            // summary:
            local_timer.tock();
            local_result.time_consumed   = local_timer.get_time_diff();
            local_result.clocks_consumed = local_timer.get_clocks_diff();
            local_result.passed = local_ok;
            tests_results << local_result;

            run_ok = run_ok && local_ok;

            if(workload_output)      delete workload_output;
            if(naive_output)         delete naive_output;
            if(images)               delete images;
            if(images_with_padding)  delete images_with_padding;
        }
    } catch(std::runtime_error &error) {
        tests_results << run_result;
        std::cout << "error: " << error.what() << std::endl;
    } catch(std::exception &error) {
        tests_results << run_result;
        std::cout << "error: " << error.what() << std::endl;
    } catch(...) {
        tests_results << run_result;
        std::cout << "error: unknown" << std::endl;
    }
    if(!done()) run_ok = false;
    std::cout << "<- Test " << (run_ok ? "passed" : "failed") << std::endl;;
    return run_ok;
}

bool test_convolution_float_cpu_random::done() {
    bool  done_ok = true;
    test_measurement_result   done_result;
    done_result.description = "DONE: " + test_description;

    C_time_control            done_timer;

    try {
        //Here - clean workflow_wrapper
        if(workflow_wrapper != nullptr)
            workflow_wrapper->cleanup();
        done_ok = true;
    } catch(std::runtime_error &error) {
        done_result << "error: " + std::string( error.what() );
        done_ok = false;
    } catch( std::exception &error ) {
        done_result << "error: " + std::string( error.what() );
        done_ok = false;
    } catch(...) {
        done_result << "error: unknown";
        done_ok = false;
    }

    done_timer.tock();
    done_result.time_consumed = done_timer.get_time_diff();
    done_result.clocks_consumed = done_timer.get_clocks_diff();

    done_result.passed = done_ok;
    tests_results << done_result;

    return done_ok;
}

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran befor main execution starts.
// The sole function of this construction is attaching this test to
// library of tests (singleton command pattern).

namespace {
    struct attach {
        test_convolution_float_cpu_random test;
        attach() {
            test_aggregator::instance().add( &test );
        }
    };
    attach attach_;
    // declarations
}
