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
#include "tester/common/workflows_for_tests.h"  // If test needs a workflow definition
#include "tester/common/test_common_tools.h"
#include "common/time_control.h"

#include <algorithm>
#include <random>
#include <climits>

class test_softmax_loss_cpu_random : public test_base {
private:
    tested_device*            current_tested_device;
    nn_device_interface_0_t*  di;

    bool init();
    bool done();
    void cleanup();

    void cpu_layer_softmax_loss( nn::data<float>&   images,
                                 nn::data<int32_t>& labels,
                                 nn::data<float>    *&naive_softmax,
                                 nn::data<float>    *&naive_loss
                                 );

    // If test needs a workflow definition
    workflows_for_tests_base    *workflow_wrapper;
    nn_workflow_t               *workflow;

    // Add current test specific variables
    uint32_t softmax_size;

public:
    test_softmax_loss_cpu_random() { test_description = "softmax loss float cpu random"; };
    ~test_softmax_loss_cpu_random() {};
    bool run();
};

void test_softmax_loss_cpu_random::cpu_layer_softmax_loss( nn::data<float>&   images,
                                                           nn::data<int32_t>& labels,
                                                           nn::data<float>*&  naive_softmax,
                                                           nn::data<float>*&  naive_loss
                                                           ) {

    const uint32_t size_x = static_cast<uint32_t>(images.size[0]),
    size_n = static_cast<uint32_t>(images.size[1]);

    naive_softmax = new nn::data<float>( size_x, size_n );
    if(naive_softmax == nullptr)   throw std::runtime_error("unable to create naive_softmax");

    naive_loss = new nn::data<float>( 1 );
    if(naive_loss == nullptr)   throw std::runtime_error("unable to create naive_loss");

    naive_loss->at(0) = 0;
    float sum = 0;

    for( uint32_t batch = 0; batch < size_n; ++batch ) {
        float max_val = 0;

        // find max elemnt (in single batch)
        for( uint32_t naive_softmax_element = 0; naive_softmax_element < size_x; ++naive_softmax_element )
            max_val = (std::max)(max_val, images.at( naive_softmax_element++, batch ));

        // subtract max_elem form all inputs (in single batch)
        for( uint32_t naive_softmax_element = 0; naive_softmax_element < size_x; ++naive_softmax_element )
            naive_softmax->at( naive_softmax_element, batch ) = images.at( naive_softmax_element, batch ) - max_val;

        // softmax
        for( uint32_t naive_softmax_element = 0; naive_softmax_element < size_x; ++naive_softmax_element ) {
            sum += exp( images.at( naive_softmax_element, batch ) );
            naive_softmax->at( naive_softmax_element, batch ) = exp( images.at( naive_softmax_element, batch ) );
        }
        sum = 1.0f / sum;
        for( uint32_t naive_softmax_element = 0; naive_softmax_element < size_x; ++naive_softmax_element )
            naive_softmax->at( naive_softmax_element, batch ) *= sum;

        naive_loss->at(0) -= std::log( std::max(naive_softmax->at(labels.at(batch), batch) , std::numeric_limits<float>::min() ));
    }

    naive_loss->at(0) /= size_n;
}

bool test_softmax_loss_cpu_random::init() {
    bool  init_ok = true;
    test_measurement_result   init_result;
    init_result.description = "INIT: " + test_description;

    C_time_control            init_timer;

    try {
        if( devices != nullptr ) {
            current_tested_device = devices->get( "device_cpu" + dynamic_library_extension );
            di = current_tested_device->get_device_interface();
        } else  throw std::runtime_error( std::string( "Can't find aggregator of devices" ) );

        // TODO: here code of test initiation:
        // If test needs a workflow definition

        workflow_wrapper = workflows_for_tests::instance().get( "workflow_for_testing_softmax_loss_float_random" );
        workflow = workflow_wrapper->init_test_workflow( di );
        if( workflow == nullptr )  throw std::runtime_error( "Workflow has not been initialized" );

        softmax_size = workflow->input[0]->output_format->format_1d.size[0];

        // END test initiation
        init_ok = true;
    } catch( std::runtime_error &error ) {
        init_result << "error: " + std::string( error.what() );
        init_ok = false;
    } catch( std::exception &error ) {
        init_result << "error: " + std::string( error.what() );
        init_ok = false;
    } catch( ... ) {
        init_result << "unknown error";
        init_ok = false;
    }

    init_timer.tock();
    init_result.time_consumed = init_timer.get_time_diff();
    init_result.clocks_consumed = init_timer.get_clocks_diff();
    init_result.passed = init_ok;

    tests_results << init_result;

    return init_ok;
}

bool test_softmax_loss_cpu_random::run() {
    bool  run_ok = true;
    test_measurement_result   run_result;
    run_result.description = "RUN SUMMARY: " + test_description;

    C_time_control  run_timer;

    std::cout << "-> Testing: " << test_description << std::endl;

    try {
        if( !init() ) throw std::runtime_error( "init() returns false so can't run test" );
        run_timer.tick();   //start time measurement
        run_result << std::string( "run test with " + current_tested_device->get_device_description() );

        NN_WORKLOAD_DATA_TYPE input_formats[]  = { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH };
        NN_WORKLOAD_DATA_TYPE output_formats[] = { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH };

        for( auto batch : {1, 8, 48 } ) {
            // ---------------------------------------------------------------------------------------------------------
            {   // simple sample pattern of test with time measuring:
                bool local_ok = true;
                test_measurement_result local_result;
                local_result.description = "RUN PART: (batch " + std::to_string( batch ) + ") execution of " + test_description;
                C_time_control  local_timer;
                // begin local test

                auto images = new nn::data<float>( softmax_size, batch );
                if(images == nullptr)   throw std::runtime_error("unable to create images nn::data for batch = " +std::to_string(batch));

                auto labels = new nn::data<int32_t>( batch );
                if(labels == nullptr)   throw std::runtime_error("unable to create labels nn::data for batch = " +std::to_string(batch));

                auto workload_output = new nn::data<float>( softmax_size, batch );
                if(workload_output == nullptr)   throw std::runtime_error("unable to create workload_output for batch = " +std::to_string(batch));

                auto workload_output_loss = new nn::data<float>( batch );
                if(workload_output_loss == nullptr)   throw std::runtime_error("unable to create workload_output_loss for batch = " +std::to_string(batch));

                nn_data_populate( workload_output, 0.0f );
                nn_data_populate( workload_output_loss, 0.0f );
                nn_data_populate( images, -255.0f, 255.0f );

                {
                    std::mt19937 int_gen(6);
                    int32_t* buf = (int32_t*)labels->buffer;
                    for(uint32_t i = 0 ; i <labels->count(); i++, buf++ )
                        *buf = int_gen() % batch;
                }

                nn_workload_t *workload = nullptr;
                nn_data_t *input_array[2] = { images, labels };
                nn::data<float> *output_array_cmpl[2] = { nn::data_cast<float, 0>(workload_output), nn::data_cast<float, 0>(workload_output_loss) };

                // TO DO fix conversion layers
                // auto status = di->workflow_compile_function( &workload, di->device, workflow, input_formats, output_formats, batch );
                // if( !workload ) throw std::runtime_error( "workload compilation failed for batch = " + std::to_string( batch )
                //                                            + " status: " + std::to_string( status ) );

                // TO DO fix conversion layers (in compile)
                //di->workload_execute_function( workload, reinterpret_cast<void**>(input_array), reinterpret_cast<void**>(output_array_cmpl), &status );

                nn::data<float>* naive_softmax = nullptr, *naive_loss = nullptr;

                cpu_layer_softmax_loss( *images, *labels, naive_softmax, naive_loss );

                // blind test
                //local_ok = compare_data(workload_output, naive_softmax);
                //local_ok = local_ok && compare_data(workload_output_loss, naive_loss);

                // end of local test
                // summary:
                local_timer.tock();
                local_result.time_consumed = local_timer.get_time_diff();
                local_result.clocks_consumed = local_timer.get_clocks_diff();
                local_result.passed = local_ok;
                tests_results << local_result;

                run_ok = run_ok && local_ok;

                if( images )               delete images;
                if( labels )               delete labels;
                if( workload_output )      delete workload_output;
                if( workload_output_loss ) delete workload_output_loss;
                if( naive_softmax )        delete naive_softmax;
                if( naive_loss )        delete naive_loss;
                if( workload )             delete workload;

            } // The pattern, of complex instruction above, can be multiplied
            // END of run tests
            // ---------------------------------------------------------------------------------------------------------
        }
    } catch( std::runtime_error &error ) {
        run_result << "error: " + std::string( error.what() );
        run_ok = false;
    } catch( std::exception &error ) {
        run_result << "error: " + std::string( error.what() );
        run_ok = false;
    } catch( ... ) {
        run_result << "unknown error";
        run_ok = false;
    }

    run_timer.tock();
    run_result.time_consumed = run_timer.get_time_diff();
    run_result.clocks_consumed = run_timer.get_clocks_diff();

    run_result.passed = run_ok;
    tests_results << run_result;
    if( !done() ) run_ok = false;
    std::cout << "<- Test " << (run_ok ? "passed" : "failed") << std::endl;
    return run_ok;
}

bool test_softmax_loss_cpu_random::done() {
    bool  done_ok = true;
    test_measurement_result   done_result;
    done_result.description = "DONE: " + test_description;

    C_time_control            done_timer;

    try {
        // TODO: here clean up after the test
        // If the test used definition of workflow:
        if( workflow_wrapper != nullptr ) workflow_wrapper->cleanup();

        // END of cleaning
        done_ok = true;
    } catch( std::runtime_error &error ) {
        done_result << "error: " + std::string( error.what() );
        done_ok = false;
    } catch( std::exception &error ) {
        done_result << "error: " + std::string( error.what() );
        done_ok = false;
    } catch( ... ) {
        done_result << "unknown error";
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
        test_softmax_loss_cpu_random test;
        attach() {
            test_aggregator::instance().add( &test );
        }
    };
    attach attach_;
}