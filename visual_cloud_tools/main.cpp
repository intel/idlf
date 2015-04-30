﻿/*
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
#include <regex>


// OS-specific constants & functions
// needed Linux APIs missing from other OSes should be emulated
#if defined _WIN32
#   include "os_windows.h"
#else
#   include "os_linux.h"
#endif

#include "FreeImage_wraps.h"
#include "workflow_builder.h"
#include "time_control.h"
#include "report_maker.h"
#include "nn_device_api.h"
#include "nn_data_tools.h"

// returns list of files (path+filename) from specified directory
std::vector<std::string> get_directory_contents(std::string images_path) {
    std::vector<std::string> result;
    if(DIR *folder = opendir(images_path.c_str())) {
        dirent *folder_entry;
        const auto image_file = std::regex(".*\\.(jpe?g|png|bmp|gif|j2k|jp2|tiff)");
        while(folder_entry = readdir(folder))
            if(std::regex_match(folder_entry->d_name, image_file) && is_regular_file(images_path, folder_entry) )
                result.push_back(images_path+ "/" +folder_entry->d_name);
        closedir(folder);
    }
    return result;
}

// RAII for library; throws runtime_error when fails
struct scoped_library {
    std::string name;
    void *handle;
public:
    scoped_library(std::string arg_name) : name(arg_name), handle(dlopen(name.c_str(), RTLD_LAZY)) {
        if(!handle) throw std::runtime_error(std::string("failed to open '")+name+"' device");
    }
    ~scoped_library() {
        dlclose(handle);
    }
    template<typename T_type> T_type *dlsym(std::string symbol) {
        if(void *sym=::dlsym(handle, symbol.c_str()))
            return reinterpret_cast<T_type*>(sym);
        else throw std::runtime_error(std::string("unable to get symbol '")+symbol+"' from device '"+name+"'");
    }
};

// RAII for device 
class scoped_device {
    scoped_library              &library_;
    decltype(nn_device_load)    *load_;
    decltype(nn_device_unload)  *unload_;
    friend class scoped_interface_0;
public:
    nn_device_description_t description;
    scoped_device(scoped_library &library) 
    : library_(library)
    , load_(  library_.dlsym<decltype(nn_device_load)>("nn_device_load"))
    , unload_(library_.dlsym<decltype(nn_device_unload)>("nn_device_unload")) {
        if(0!=load_(&description)) throw std::runtime_error(std::string("failed to load device '")+library_.name+"'");
    }
    ~scoped_device() {
        unload_();
    }
};

// RAII for interface
class scoped_interface_0 : public nn_device_interface_0_t {
    scoped_library                         &library_;
    decltype(nn_device_interface_open)     *open_;
    decltype(nn_device_interface_close)    *close_;
public:
    scoped_interface_0(scoped_device &device)
    : library_(device.library_)
    , open_( library_.dlsym<decltype(nn_device_interface_open)>("nn_device_interface_open"))
    , close_(library_.dlsym<decltype(nn_device_interface_close)>("nn_device_interface_close"))
    {
        if(0!=open_(0, this)) throw std::runtime_error(std::string("failed to open interface ")+std::to_string(version)+" from device '"+library_.name+"'");
    }
    ~scoped_interface_0() {
        close_(this);
    }
};

// parses parameters stored as vector of strings and insersts them into map
void parse_parameters(std::map<std::string, std::string> &config, std::vector<std::string> &input) {
    std::basic_regex<char> regex[] = {
          std::regex("(--|-|\\/)([a-z][a-z-_]*)=((.*)|)")   // key-value pair
        , std::regex("(--|-|\\/)([a-z][a-z-_]*)")           // key only
        , std::regex(".+")                                  // this must be last in regex[]; what is left is a filename
    };
    for(std::string &in : input) {
        std::match_results<const char *> result;
        const auto regex_count = sizeof(regex)/sizeof(regex[0]);
        const auto regex_last = regex_count-1;
        for(size_t index=0; index<regex_count; ++index)
            if(std::regex_match(in.c_str(), result, regex[index])) {
                if(index==regex_last) {
                    if(config.find("input")==std::end(config))
                        config["input"] = result[2];
                } else {
                    if(config.find(result[2])==std::end(config)) {
                        config[result[2]] = result[3];
                        break;
                    }
                }
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
        can be caffenet_float or caffenet_int16
    --input=<directory>
        path to directory that contains images to be classfied
    --config=<name>
        file name of config file containing additional parameters
        command line parameters take priority over config ones

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

            auto not_found = std::end(config);
            if(config.find("device")==not_found) config["device"]="device_cpu";
            if(config.find("batch") ==not_found) config["batch"]="48";
            if(config.find("model") ==not_found) config["model"]="caffenet_float";
            if(config.find("input") ==not_found) throw std::runtime_error("missing input directory; run without arguments to get help");
            if(config.find("loops") ==not_found) config["loops"]="1";
        }

        // load images from input directory
        auto images_list = get_directory_contents(config["input"]);
        if(images_list.empty()) throw std::runtime_error(std::string("directory ")+config["input"]+" does not contain any images that can be processed");

        // RAII for loading library, device initialization and opening interface 0
        scoped_library      library(config["device"]+dynamic_library_extension);
        scoped_device       device(library);
        scoped_interface_0  interface_0(device);

        // get workflow builder as specified by model parameter
        auto builder = workflow_builder::instance().get(config["model"]);
        const int config_batch = std::stoi(config["batch"]);
        if(config_batch<=0) throw std::runtime_error("batch_size is 0 or negative");
        nn_workflow_t *workflow = builder->init_workflow(&interface_0, config_batch);

        // compiling workload
        NN_WORKLOAD_DATA_TYPE input_format = NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH;
        NN_WORKLOAD_DATA_TYPE output_format = NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH;
        nn_workload_t *workload = nullptr;
        C_time_control timer;
        auto status = interface_0.workflow_compile_function(&workload, interface_0.device, workflow, &input_format, &output_format, config_batch);
        timer.tock();
        if(!workload) throw std::runtime_error("workload compilation failed");
        std::cout << "workload compiled in " << timer.time_diff_string() <<" [" <<timer.clocks_diff_string() <<"]" << std::endl;

        // 1000 classes as a workload output
        auto workload_output = new nn::data<float, 2>(1000, config_batch);
        nn::data<float> *output_array_cmpl[1] ={ nn::data_cast<float, 0>(workload_output) };


        uint16_t                   image_counter = 0;     //numbering images within single batch
        bool                       start_batch = false;

        C_report_maker report(config_batch);

        std::cout << "recognizing " << images_list.size() << " image(s)" << std::endl;

        auto images_list_iterator = images_list.begin();
        auto images_list_end = images_list.end();

        while(images_list_iterator!=images_list_end) {

            auto diff_itr = images_list_end - images_list_iterator < config_batch 
                          ? images_list_end - images_list_iterator 
                          : config_batch;

            std::vector<std::string>   batch_images (images_list_iterator,images_list_iterator+diff_itr);

            images_list_iterator+=diff_itr;

            nn::data<float, 4> *images = nullptr;
            images = nn_data_load_from_image_list(&batch_images, builder->get_img_size(), builder->image_process, config_batch, builder->RGB_order);

            if(images) {
                images_recognition_batch_t  temp_report_recognition_batch;
                nn_data_t *input_array[1] ={images};
                {
                    NN_API_STATUS  status;
                    C_time_control timer;
                    auto loops = std::stoi(config["loops"]);
                    for(size_t i=0; i <loops; ++i)
                    {
                        interface_0.workload_execute_function(workload,reinterpret_cast<void**>(input_array),reinterpret_cast<void**>(output_array_cmpl),&status);
                    }
                    timer.tock();
                    temp_report_recognition_batch.time_of_recognizing = timer.get_time_diff()/loops;
                    temp_report_recognition_batch.clocks_of_recognizing = timer.get_clocks_diff()/loops;
                }

                delete images;

                float* value_cmpl = reinterpret_cast<float*>(workload_output->buffer);

                auto batch_images_iterator = batch_images.begin();

                for(auto b = 0u; b < batch_images.size(); ++b) {

                    image_recognition_item_t    temp_report_recognition_item;

                    recognition_state_t         temp_recognition_state;
                    std::map <float,int>       output_values;

                    temp_report_recognition_item.recognitions.clear();
                    temp_report_recognition_item.recognized_image = *batch_images_iterator++;

                    for(int index = 0; index < 1000; ++index) {
                        output_values.insert(std::make_pair(value_cmpl[index],index));
                        temp_report_recognition_item.nnet_output.push_back(value_cmpl[index]);
                    }

                    temp_report_recognition_item.wwid = temp_report_recognition_item.recognized_image.find('[') != std::string::npos
                        ? temp_report_recognition_item.recognized_image.substr(temp_report_recognition_item.recognized_image.find('[') + 1,9) 
                        : "n000000000";
                    auto iterator = --output_values.end();
                    for(int i = 1; i < 6 && iterator != output_values.end(); ++i)
                    {
                        temp_recognition_state.label    = builder->labels[iterator->second];
                        temp_recognition_state.wwid     = builder->wwids[iterator->second];
                        temp_recognition_state.accuracy = iterator->first;
                        temp_report_recognition_item.recognitions.push_back(temp_recognition_state);
                        --iterator;
                    }
                    temp_report_recognition_batch.recognized_images.push_back(temp_report_recognition_item);
                    output_values.clear();
                    value_cmpl += 1000;
                }
                batch_images.clear();
                report.recognized_batches.push_back(temp_report_recognition_batch);
                temp_report_recognition_batch.recognized_images.clear();
            }
        }
        report.print_to_html_file("index.html", "Results of recognition");
        system((show_HTML_command+"index.html").c_str());
    return 0;
    }
    catch(std::runtime_error &error) {
        std::cout << "error: " << error.what() << std::endl;
    }
    catch(...) {
        std::cout << "unknown error" << std::endl;
    }
}
