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
#include "common/time_control.h"
#include <numeric>

class test_caffe_float_workload_cpu_time: public test_base
{
private:
    tested_device*            current_tested_device = nullptr;
    nn_device_interface_0_t*  di = nullptr;
    uint32_t                  loops = 100;
    uint16_t                  img_size = 227;



    bool init();
    bool done();
    void cleanup();

    workflows_for_tests_base    *workflow_wrapper = nullptr;
    nn_workflow_t               *workflow = nullptr;

public:
    test_caffe_float_workload_cpu_time() { test_description = "caffe float workload cpu time"; };
    ~test_caffe_float_workload_cpu_time() {};
    bool run();
};

bool test_caffe_float_workload_cpu_time::init()
{
    bool  init_ok = true;
    test_measurement_result   init_result;
    init_result.description = "INIT: " + test_description;

    C_time_control            init_timer;

    try {
        if(devices!=nullptr)  {
            current_tested_device = devices->get("device_cpu"+dynamic_library_extension);
            di = current_tested_device->get_device_interface();
        }
        else  throw std::runtime_error("Can't find aggregator of devices");

        // HERE init workflow
        workflow_wrapper = workflows_for_tests::instance().get("caffenet_float_learned");
        workflow = workflow_wrapper->init_test_workflow(di);
        if(workflow == nullptr)  throw std::runtime_error("Workflow has not been initialized");

        init_ok = true;
    }
    catch(std::runtime_error &error) {
        init_result << "error: " + std::string(error.what());
        init_ok = false;
    }
    catch(...) {
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

bool test_caffe_float_workload_cpu_time::run()
{
    bool  run_ok = true;
    test_measurement_result   run_result;
    run_result.description = "RUN SUMMARY: " + test_description;

    C_time_control  run_timer;

    std::cout << "-> Testing: " << test_description << std::endl;

    try {
        if(!init()) throw std::runtime_error("error: init() returns false so can't run test");
        run_timer.tick();   //start time measurement
        run_result << std::string("run test with " + current_tested_device->get_device_description());
        // ---------------------------------------------------------------------------------------------------------
        // TODO: here test code
        //{   // BKM pattern of test with time measuring:
        //    bool local_ok=true;
        //    test_measurement_result local_result;
        //    local_result.description = "RUN PART: (name part) of " + test_description;
        //    C_time_control  local_timer;
        //    // begin local test

        //    // end of local test
        //    // summary:
        //    local_timer.tock();
        //    local_result.time_consumed = local_timer.time_diff_string();
        //    local_result.clocks_consumed = local_timer.get_clocks_diff();
        //    tests_results << local_result;
        //} // The pattern, of complex instruction above, can be multiplied
        for(uint16_t batch :{1,8,48})
        {

            std::vector<uint64_t>     time_diffs;
            std::vector<uint64_t>     clock_diffs;

            nn::data<float,4>        *images = new nn::data<float,4>(img_size,img_size,3,batch);
            nn_data_populate(nn::data_cast<float,0>(images),0.0f,255.0f);
            nn_data_t *input_array[1] ={images};

            auto workload_output = new nn::data<float, 2>(1000, batch);
            nn::data<float> *output_array_cmpl[1] ={ nn::data_cast<float, 0>(workload_output) };

            nn_workload_t             *workload = nullptr;

            // compiling workload
            NN_WORKLOAD_DATA_TYPE input_format = NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH;
            NN_WORKLOAD_DATA_TYPE output_format = NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH;


            auto status = di->workflow_compile_function(&workload,di->device,workflow,&input_format,&output_format,batch);
            if(!workload) throw std::runtime_error("workload compilation failed for batch = " + std::to_string(batch)
                                                   + " status: " + std::to_string(status));

            test_measurement_result local_result;
            local_result.description = "RUN PART: (batch " + std::to_string(batch)+") execution of " + test_description;
            local_result.loops = loops;

            // begin local test
            for(auto i = 0; i< loops; ++i)
            {
                NN_API_STATUS   status;
                C_time_control  loop_timer;
                di->workload_execute_function(workload,reinterpret_cast<void**>(input_array),reinterpret_cast<void**>(output_array_cmpl),&status);
                loop_timer.tock();
                time_diffs.push_back(loop_timer.get_time_diff()/batch);
                clock_diffs.push_back(loop_timer.get_clocks_diff()/batch);
            }

            // end of local test
            // summary:
            uint64_t  min_value = *std::min_element(time_diffs.begin(),time_diffs.end());
            local_result.time_consumed = std::accumulate(time_diffs.begin(),time_diffs.end(),0.0)/time_diffs.size();
            local_result.time_consumed_min = min_value;
            local_result.time_consumed_max = *std::max_element(time_diffs.begin(),time_diffs.end());

            local_result << std::string("note: The shortest time for one image obtained from the chrono: "
                                        + C_time_control::time_diff_string(min_value));
            local_result << std::string("note: Values of time's and clock's were divided by current value of batch: "+std::to_string(batch));

            local_result.clocks_consumed = std::accumulate(clock_diffs.begin(),clock_diffs.end(),0.0)/clock_diffs.size();
            local_result.clocks_consumed_min = *std::min_element(clock_diffs.begin(),clock_diffs.end());
            local_result.clocks_consumed_max = *std::max_element(clock_diffs.begin(),clock_diffs.end());

            tests_results << local_result;
            if(images != nullptr) delete images;
            if(workload_output != nullptr) delete workload_output;
            if(workload != nullptr) di->workload_delete_function(workload);
        }
        // ---------------------------------------------------------------------------------------------------------
        run_ok = true;
    }
    catch(std::runtime_error &error) {
        run_result << "error: " + std::string(error.what());
        run_ok = false;
    }
    catch(...) {
        run_result << "error: unknown";
        run_ok = false;
    }

    run_timer.tock();
    run_result.time_consumed = run_timer.get_time_diff();
    run_result.clocks_consumed = run_timer.get_clocks_diff();

    run_result.passed = run_ok;
    tests_results << run_result;
    if (!done()) run_ok=false;
    std::cout << "<- Test " << (run_ok ? "passed" : "failed") << std::endl;;
    return run_ok;
}

bool test_caffe_float_workload_cpu_time::done()
{
    bool  done_ok = true;
    test_measurement_result   done_result;
    done_result.description = "DONE: " + test_description;

    C_time_control            done_timer;

    try {
        //Here - clean workflow_wrapper
        if(workflow_wrapper!=nullptr)
           workflow_wrapper->cleanup();
        done_ok = true;
    }
    catch(std::runtime_error &error) {
        done_result << "error: " + std::string(error.what());
        done_ok = false;
    }
    catch(...) {
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
        test_caffe_float_workload_cpu_time test;
        attach() {
            test_aggregator::instance().add(&test);
        }
    };
    attach attach_;
}