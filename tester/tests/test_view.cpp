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
#include "device/common/nn_workload_data.h"
#include "device/cpu/api_internal/data_helper.h"

#include <random>

class test_view : public test_base {
private:
    tested_device*            current_tested_device;
    nn_device_interface_0_t*  di;

    bool init();
    bool done();
    void cleanup();

    nn::data<float>* naive_view( nn::data<float> &work_item, nn_workload_data_coords_t& begin_coords, nn_workload_data_coords_t& end_coords );
    // Add current test specific variables

public:
    test_view() { test_description = "view float cpu random"; };
    ~test_view() {};
    bool run();
};

bool test_view::init() {
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

nn::data<float>* test_view::naive_view( nn::data<float> &work_item, nn_workload_data_coords_t& begin_coords, nn_workload_data_coords_t& end_coords ) {

        uint32_t end_x   =   end_coords.t[1] + 1,
                 end_y   =   end_coords.t[2] + 1,
                 end_z   =   end_coords.t[3] + 1,
                 end_n   =   end_coords.t[0] + 1,
                 begin_x = begin_coords.t[1],
                 begin_y = begin_coords.t[2],
                 begin_z = begin_coords.t[3],
                 begin_n = begin_coords.t[0];

        auto output = new nn::data<float>( end_z - begin_z,
                                           end_x - begin_x,
                                           end_y - begin_y,
                                           end_n - begin_n );
        if(output == nullptr)   throw std::runtime_error("unable to create output");

        for(uint32_t n = begin_n; n < end_n; ++n)
            for(uint32_t z = begin_z; z < end_z; ++z)
                for( uint32_t y = begin_y; y < end_y; ++y )
                    for( uint32_t x = begin_x; x < end_x; ++x ) {
                        output->at(
                            z - begin_z,
                            x - begin_x,
                            y - begin_y,
                            n - begin_n
                            ) = work_item.at(z, x, y, n);
                    }

    return output;
}

bool test_view::run() {
    bool  run_ok = true;
    test_measurement_result   run_result;
    run_result.description = "RUN SUMMARY: " + test_description;

    C_time_control  run_timer;

    std::cout << "-> Testing: " << test_description << std::endl;
    try {
        if( !init() ) throw std::runtime_error( "init() returns false so can't run test" );
        run_timer.tick();   //start time measurement
        run_result << std::string( "run test with " + current_tested_device->get_device_description() );

        NN_WORKLOAD_DATA_TYPE input_format  = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
        NN_WORKLOAD_DATA_TYPE output_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;

        std::mt19937 generator( 1 );
        std::uniform_int_distribution<uint32_t> distribution( 0, 56/2 );

        auto compare_data = [](nn::workload_data<nn::layout_f32>& item, nn::data<float>& ref_item) {
                float relative_error_threshold = 1e-3f,
                      absolute_error_threshold = 1e-6f,
                      absoulte_error_limit     = 1e-4f;

                uint32_t size_n = item.get_length(0),
                         size_x = item.get_length(1),
                         size_y = item.get_length(2),
                         size_z = item.get_length(3);

                for(uint32_t n = 0; n < size_n; ++n)
                    for(uint32_t z = 0; z < size_z; ++z)
                        for( uint32_t y = 0; y < size_y; ++y )
                            for( uint32_t x = 0; x < size_x; ++x ) {
                                float workload_val = item.at(n, x, y, z, 0, 0);
                                float ref_val      = ref_item.at(z, x, y, n);

                                 if( fabs(workload_val) < absoulte_error_limit) {
                                    if(fabs( workload_val - ref_val ) > absolute_error_threshold) {
                                        return false;
                                    }
                                } else
                                    if(fabs(workload_val - ref_val) / fabs(ref_val) > relative_error_threshold)
                                        return false;
                            }
            return true;
        };

        for( uint32_t batch : { 1, 8, 48 } ) {
            // simple sample pattern of test with time measuring:
            bool local_ok = true;
            test_measurement_result local_result;
            local_result.description = "RUN PART: (batch " + std::to_string( batch ) + ") execution of " + test_description;
            C_time_control  local_timer;

            for(uint32_t size_x : { 5,16,56 }) {
                for(uint32_t size_y : { 5,16,56 }) {
                    for(uint32_t size_z : { 1,8,16 }) {
                        // ---------------------------------------------------------------------------------------------------------
                        // begin local test
                        auto input = new nn::data<float>(size_z,size_x,size_y,batch);
                        if(input == nullptr)   throw std::runtime_error("unable to create input nn::data for batch = " +std::to_string(batch));

                        nn_data_populate(input,-100.0f,100.0f);
                        auto wrkld_data = new nn::workload_data<nn::layout_f32>(input->buffer,
                                                                                {batch,size_x,size_y,size_z,1,1},
                                                                                 nn::data_helper_layout_lookup_zxynpq<float>()
                                                                               );
                        if(wrkld_data == nullptr) {
                            delete input;
                            throw std::runtime_error("unable to create wrkld_data for batch = " +std::to_string(batch));
                        }

                        nn_workload_data_coords_t* view_begin_coords,*view_end_coords;
                        { // create random view
                            view_begin_coords = new nn_workload_data_coords_t{
                                distribution(generator) % batch,
                                distribution(generator) % size_x,
                                distribution(generator) % size_y,
                                distribution(generator) % size_z,
                                0,
                                0
                            };
                            if(view_begin_coords == nullptr) {
                                delete input;
                                delete wrkld_data;
                                throw std::runtime_error("unable to create view_begin_coords for batch = " +std::to_string(batch));
                            }

                            view_end_coords  = new nn_workload_data_coords_t{
                                distribution(generator) % batch,
                                distribution(generator) % size_x,
                                distribution(generator) % size_y,
                                distribution(generator) % size_z,
                                0,
                                0
                            };
                            if(view_end_coords == nullptr) {
                                delete input;
                                delete wrkld_data;
                                delete view_begin_coords;
                                throw std::runtime_error("unable to create view_end_coords for batch = " +std::to_string(batch));
                            }

                            for(int i = 0 ; i <= 4 ; ++i)
                                if(view_begin_coords->t[i] > view_end_coords->t[i]) {
                                    std::swap(view_begin_coords->t[i],view_end_coords->t[i]);
                                }
                        }

                        // create view
                        auto workload_output = new nn::workload_data<nn::layout_f32>(*wrkld_data,*view_begin_coords,*view_end_coords);
                        if(workload_output == nullptr) {
                            delete input;
                            delete wrkld_data;
                            delete view_begin_coords;
                            delete view_end_coords;
                            delete workload_output;
                            throw std::runtime_error("unable to create workload_output nn::workload_data for batch = " +std::to_string(batch));
                        }

                        // naive view
                        auto naive_output = naive_view(*input,*view_begin_coords,*view_end_coords);

                        local_ok = compare_data(*workload_output,*naive_output);

                        if(input)            delete input;
                        if(workload_output)  delete workload_output;
                        if(naive_output)     delete naive_output;
                        if(view_begin_coords)delete view_begin_coords;
                        if(view_end_coords)  delete view_end_coords;
                        if(wrkld_data)       delete wrkld_data;
                        // END of run tests
                        // ---------------------------------------------------------------------------------------------------------
                    } // The pattern, of complex instruction above, can be multiplied
                }
            }
            // end of local test
            // summary:
            local_timer.tock();
            local_result.time_consumed = local_timer.get_time_diff();
            local_result.clocks_consumed = local_timer.get_clocks_diff();
            local_result.passed = local_ok;
            tests_results << local_result;

            run_ok = run_ok && local_ok;
        }
    }
    catch(std::runtime_error &error) {
        run_result << "error: " + std::string(error.what());
        run_ok = false;
    }
    catch(std::exception &error) {
        run_result << "error: " + std::string(error.what());
        run_ok = false;
    }
    catch(...) {
        run_result << "unknown error";
        run_ok = false;
    }
    run_timer.tock();
    run_result.time_consumed = run_timer.get_time_diff();
    run_result.clocks_consumed = run_timer.get_clocks_diff();

    run_result.passed = run_ok;
    tests_results << run_result;
    if(!done()) run_ok = false;
    std::cout << "<- Test " << (run_ok ? "passed" : "failed") << std::endl;
    return run_ok;
}

bool test_view::done() {
    bool  done_ok = true;
    test_measurement_result   done_result;
    done_result.description = "DONE: " + test_description;

    C_time_control            done_timer;

    try {
        // TODO: here clean up after the test
        // If the test used definition of workflow:

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
        test_view test;
        attach() {
            test_aggregator::instance().add( &test );
        }
    };
    attach attach_;
}
