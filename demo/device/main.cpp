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

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>


// OS & compiler-specific constants,  functions & workarounds
// needed Linux APIs missing from other OSes should be emulated
#if defined _WIN32
#   include "common/os_windows.h"
#else
#   include "common/os_linux.h"
#endif

#include "common/FreeImage_wraps.h"
#include "workflow_builder.h"
#include "workload_execute.h"
#include "common/time_control.h"
#include "demo/common/report_maker.h"
#include "device/api/nn_device_api.h"
#include "common/nn_data_tools.h"



// parses parameters stored as vector of strings and insersts them into map
void parse_parameters(std::map<std::string, std::string> &config, std::vector<std::string> &input) {
    std::regex regex[] = {
          std::regex("(--|-|\\/)([a-z][a-z\\-_]*)=(.*)")    // key-value pair
        , std::regex("(--|-|\\/)([a-z][a-z\\-_]*)")         // key only
        , std::regex(".+")                                  // this must be last in regex[]; what is left is a filename
    };
    for(std::string &in : input) {
        std::cmatch result;
        const auto regex_count = sizeof(regex)/sizeof(regex[0]);
        const auto regex_last = regex_count-1;
        for(size_t index=0; index<regex_count; ++index)
            if(std::regex_match(in.c_str(), result, regex[index])) {
                std::string key   = index==regex_last ? std::string("input") : result[2];
                std::string value = result[index==regex_last ? 0 : 3];
                config.insert(std::make_pair(key, value));
                break;
            }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    try {
        if(argc<=1) {
            const std::string path(argv[0]);
            std::cout << "usage: " << path.substr(path.find_last_of("/\\") + 1) <<
R"_help_( <parameters> input_dir

<parameters> include:
    --device=<name>
        name of dynamic library (without suffix) with computational device
        to be used for demo
    --batch=<value>
        size of group of images that are classified together;  large batch
        sizes have better performance
    --model=<name>
        name of network model that is used for classfication
        can be: caffenet_float,  caffenet_int16 or lenet_float
    --training
        Run a training mode of selected model (default is classification)
    --input=<directory>
        path to directory that contains images to be classfied
    --config=<name>
        file name of config file containing additional parameters
        command line parameters take priority over config ones
    --save_reference
        Save softmax output for each image in txt format
        Reference's filename will the same as image's name
    --use_jit_primitives
        Enable usage of faster jit primitives.
        Will work only with classification (forward primitives)
        and batch, that is multiplication of 24

If last parameters do not fit --key=value format it is assumed to be a --input.
Instead of "--" "-" or "/" can be used.
)_help_";
            return 0;
        }

        // convert argc/argv to vector of arguments
        std::vector<std::string> arg;
        for(int n=1; n<argc; ++n) arg.push_back(argv[n]);

        // parse configuration (command line and from file)
        using config_t = std::map<std::string, std::string>;
        config_t config;
        parse_parameters(config, arg);
        { // find if config file name was given from commandline
            config_t::iterator it;
            if((it = config.find("config"))!=std::end(config)){
                std::ifstream config_file(it->second);
                std::vector<std::string> config_lines;
                using istream_it = std::istream_iterator<std::string>;
                std::copy(istream_it(config_file), istream_it(), std::back_inserter(config_lines));
                parse_parameters(config, config_lines);
            }
        }
        { // validate & add defalut value for missing arguments
          // default device is device_cpu  its default batch is 48
          // if other devices (gpu or int16) are given its default batch is 32
            auto not_found = std::end(config);
            if(config.find("device")==not_found)
            {
                config["device"]="device_cpu";
                if(config.find("batch") ==not_found) config["batch"]="48";
            }
            else
            {
                if( config.find("batch") == not_found)
                {
                    config["batch"] = (config["device"] == "device_cpu" ? "48" : "32");
                }
            }
            if(config.find("model") ==not_found) config["model"]="caffenet_float";
            if(config.find("input") ==not_found) throw std::runtime_error("missing input directory; run without arguments to get help");
            if(config.find("loops") ==not_found) config["loops"]="1";
        }

        // RAII for loading library, device initialization and opening interface 0
        scoped_library      library(config["device"]+dynamic_library_extension);
        scoped_device       device(library);
        scoped_interface_0  interface_0(device);

        std::string builder_desc(config["model"]);

        // If training is to happen then builder of training should be chosen
        if( config.find("training") != std::end(config))
        {
            builder_desc += "_training";
        }
        // get workflow builder as specified by model parameter
        auto builder = workflow_builder::instance().get(builder_desc);

        const int config_batch = std::stoi(config["batch"]);

        if(config_batch<=0) throw std::runtime_error("batch_size is 0 or negative");
        nn_workflow_t *workflow = builder->init_workflow(&interface_0);
        nn_workload_t *workload = nullptr;
        C_time_control timer;
        if (config.find("use_jit_primitives") != config.end())
            interface_0.use_jit_primitives(1);
        auto status = interface_0.workflow_compile_function(
            &workload,
            interface_0.device,
            workflow,
            builder->get_input_formats(),
            builder->get_output_formats(),
            config_batch);

        timer.tock();
        if(!workload) throw std::runtime_error("workload compilation failed");
        std::cout << "workload compiled in " << timer.time_diff_string() <<" [" <<timer.clocks_diff_string() <<"]" << std::endl;

        if( config.find("training") == std::end(config))
        {
            //TODO: currently lenet model comes to use mnist database for training/testing
            // so It could be changed to recgonize digits from regular images
            if( config["model"].compare("lenet_float") == 0 )
            {
                run_mnist_classification(library,device,interface_0,workload,builder,argv,config,config_batch);
            }
            else
            {
                run_images_classification(library,device,interface_0,workload,builder,argv,config,config_batch);
            }
        }
        else
        {
            if( config["model"].compare("lenet_float") == 0 )
            {
                run_mnist_training(library,device,interface_0,workload,builder,argv,config,config_batch);
            }
            else
            {
                run_images_training(library,device,interface_0,workload,builder,argv,config,config_batch);
            }
        }
    return 0;
    }
    catch(std::runtime_error &error) {
        std::cout << "error: " << error.what() << std::endl;
        return -1;
    }
    catch(std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
    catch(...) {
        std::cout << "error: unknown" << std::endl;
        return -1;
    }
}
