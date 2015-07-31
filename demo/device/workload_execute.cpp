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
#include "workload_execute.h"
#include "common/time_control.h"
#include "demo/common/report_maker.h"
#include "device/api/nn_device_api.h"
#include "common/nn_data_tools.h"

#include <memory>
#include <cstring>

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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void run_images_classification( scoped_library      &library,
                         scoped_device       &device,
                         scoped_interface_0  &interface_0,
                         nn_workload_t *workload,
                         workflow_builder_base* builder,
                         char *argv[], 
                         std::map<std::string, std::string> &config, 
                         const int config_batch )
{
    // load images from input directory
    auto images_list = get_directory_contents(config["input"]);
    if(images_list.empty()) {
        throw std::runtime_error(std::string("directory ")+config["input"]+" does not contain any images that can be processed");
    }
    // 1000 classes as a workload output
    auto              workload_output       = new nn::data< float, 2 >( 1000, config_batch );
    nn::data< float > *output_array_cmpl[1] = { nn::data_cast< float, 0 >( workload_output ) };


    const std::string path( argv[0] );
    std::string appname( path.substr( path.find_last_of( "/\\" ) + 1 ) );

    C_report_maker report( appname, library.name, config["model"], config_batch );

    std::cout << "recognizing " << images_list.size() << " image(s)" << std::endl;
    std::cout << "loops: " << config["loops"] << std::endl;

    auto images_list_iterator = images_list.begin();
    auto images_list_end      = images_list.end();

    while( images_list_iterator != images_list_end )
    {

        auto diff_itr = images_list_end - images_list_iterator < config_batch
                        ? images_list_end - images_list_iterator
                        : config_batch;

        std::vector< std::string >   batch_images( images_list_iterator, images_list_iterator + diff_itr );

        images_list_iterator += diff_itr;

        nn::data< float, 4 > *images = nullptr;
        images = nn_data_load_from_image_list( &batch_images,
                                               builder->get_img_size(), builder->image_process, config_batch,
                                               builder->RGB_order );

        if( images )
        {
            images_recognition_batch_t temp_report_recognition_batch;
            nn_data_t                  *input_array[1] = {images};
            {
                NN_API_STATUS  status;
                C_time_control timer;
                auto           loops = std::stoi( config["loops"] );
                for( size_t i = 0; i < loops; ++i )
                {
                    interface_0.workload_execute_function( workload,
                                                           reinterpret_cast< void ** >( input_array ),
                                                           reinterpret_cast< void ** >( output_array_cmpl ),
                                                           &status );
                }
                timer.tock();
                temp_report_recognition_batch.time_of_recognizing   = timer.get_time_diff() / loops;
                temp_report_recognition_batch.clocks_of_recognizing = timer.get_clocks_diff() / loops;
            }


            delete images;

            float *value_cmpl = reinterpret_cast< float * >( workload_output->buffer );

            auto batch_images_iterator = batch_images.begin();

            for( auto b = 0u; b < batch_images.size(); ++b )
            {

                image_recognition_item_t temp_report_recognition_item;

                recognition_state_t     temp_recognition_state;
                std::map < float, int > output_values;

                temp_report_recognition_item.recognitions.clear();
                temp_report_recognition_item.recognized_image = *batch_images_iterator++;

                for( int index = 0; index < 1000; ++index )
                {
                    output_values.insert( std::make_pair( value_cmpl[index], index ) );
                    temp_report_recognition_item.nnet_output.push_back( value_cmpl[index] );
                }
                temp_report_recognition_item.wwid = temp_report_recognition_item.recognized_image.find( '[' ) !=
                                                    std::string::npos
                                                    ? temp_report_recognition_item.recognized_image.substr( temp_report_recognition_item.recognized_image.find( '[' ) + 1, 9 ) : "n000000000";
                auto iterator = --output_values.end();
                for( int i = 1; i < 6 && iterator != output_values.end(); i++ )
                {
                    temp_recognition_state.label    = builder->labels[iterator->second];
                    temp_recognition_state.wwid     = builder->wwids[iterator->second];
                    temp_recognition_state.accuracy = iterator->first;
                    temp_report_recognition_item.recognitions.push_back( temp_recognition_state );
                    --iterator;
                }

                temp_report_recognition_batch.recognized_images.push_back( temp_report_recognition_item );
                output_values.clear();
                value_cmpl += 1000;
            }
            batch_images.clear();
            report.recognized_batches.push_back( temp_report_recognition_batch );
            temp_report_recognition_batch.recognized_images.clear();
        }
    }
    interface_0.workload_delete_function( workload );
    std::string html_filename = "result_" + get_timestamp() + ".html";

    report.print_to_html_file( html_filename, "Results of recognition" );
    system( ( show_HTML_command + html_filename ).c_str() );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void run_mnist_classification( scoped_library      &library,
                         scoped_device       &device,
                         scoped_interface_0  &interface_0,
                         nn_workload_t *workload,
                         workflow_builder_base* builder,
                         char *argv[], 
                         std::map<std::string, std::string> &config, 
                         const int config_batch )
{
    std::string mnist_images(config["input"]+"/t10k-images-idx3-ubyte");
    std::string mnist_labels(config["input"]+"/t10k-labels-idx1-ubyte");

    nn::data< float, 3 > *images = nullptr;
    nn::data< char, 1 > *labels = nullptr;
    
    unsigned int num_proper_classifications = 0;
    auto         loops                      = std::stoi( config["loops"] );

    std::cout << "MNIST classification using: " << mnist_images << " and " << mnist_labels << std::endl;

    // 1. Read all images and labels
    nn_data_load_images_and_labels_from_mnist_files( images, labels, mnist_images, mnist_labels);
    // continer for batch images to be processed    
    auto images_to_be_recognized = std::unique_ptr<nn::data<float,3>>(new nn::data<float, 3>(images->size[0], images->size[1], config_batch)); 

    // MNIST is a data base of digits to there is only 10 potential classes for every recognized item
    auto workload_output = new nn::data< float, 2 >( 10, config_batch );
    nn::data< float > *output_array_cmpl[1] = { nn::data_cast< float, 0 >( workload_output ) };

    NN_API_STATUS  status;

    //2. Classifiy batches of images .   
    int rem_images = images->size[2] % config_batch;
    int total_num_images_to_classify = (images->size[2] / config_batch ) * config_batch + (rem_images != 0)*config_batch;
    for(unsigned int image_index = 0; image_index < total_num_images_to_classify; image_index+=config_batch)
    {
        unsigned int num_images_to_ignore = 0;
        // 2.1 Make nn_data containing only images to be recognized in one execute (number of images depends on batch size)
        if(image_index + config_batch <= images->size[2]) {
        std::memcpy(images_to_be_recognized->buffer,
                    ((float*)images->buffer) + image_index*images->size[0]*images->size[1],
                    config_batch*images->size[0]*images->size[1]*sizeof(float));
        } else {
            //    classify images that did not fit into full batch
            //    we will put remaining images along with previously classified to get a full batch
            //    ignore when counting number of proper classifications already classfied images (config_batch - rem_images) of classification will be ignored
            std::memcpy(images_to_be_recognized->buffer,
                        ((float*)images->buffer) +  (images->size[2] - config_batch)*images->size[0]*images->size[1],
                        config_batch*images->size[0]*images->size[1]*sizeof(float));
            num_images_to_ignore = config_batch - rem_images;
        }
        nn_data_t                  *input_array[1] = {images_to_be_recognized.get()};

        for( size_t i = 0; i < loops; ++i )
        {
            interface_0.workload_execute_function( workload,
                                                   reinterpret_cast< void ** >( input_array ),
                                                   reinterpret_cast< void ** >( output_array_cmpl ),
                                                   &status );
        }
        // Get number of proper classifications within a processed batch of images
        auto num_proper_classifications_within_batch =  [&output_array_cmpl,&labels,&image_index,&config_batch,&num_images_to_ignore]()
        {
            unsigned int num_proper_classifications_in_batch = 0;
            
            for(unsigned int batch = num_images_to_ignore; batch < config_batch; ++batch )
            {
                float best_score = output_array_cmpl[0]->at(0,batch);
                unsigned int candidate_digit = 0;
                for(uint8_t i=1; i < output_array_cmpl[0]->size[0]; ++i )
                {
                    if(best_score < output_array_cmpl[0]->at(i,batch))
                    {
                        best_score = output_array_cmpl[0]->at(i,batch);
                        candidate_digit = i; 
                    }
                }
                num_proper_classifications_in_batch += ((char)candidate_digit == labels->at(  image_index + batch - num_images_to_ignore)) ? 1 : 0;
            }
            return num_proper_classifications_in_batch; 
        };
        num_proper_classifications += num_proper_classifications_within_batch(); 
    }

    std::cout << "Results: " << num_proper_classifications << " were recognized properly out of " << images->size[2] << std::endl;
    std::cout << "error rate: " << (float)((images->size[2] - num_proper_classifications) / (float)images->size[2]) << std::endl;

    // Release containers
    delete workload_output;
    workload_output = nullptr;
    delete images;
    images = nullptr;
    delete labels;
    labels = nullptr;

    return;
}
//////////////////////////////////////////////////////////////////////////////////
void run_mnist_training( scoped_library      &library,
                         scoped_device       &device,
                         scoped_interface_0  &interface_0,
                         nn_workload_t *workload,
                         workflow_builder_base* builder,
                         char *argv[], 
                         std::map<std::string, std::string> &config, 
                         const int config_batch )
{
    std::cout << "--->TRENING" << std::endl;
    std::string mnist_images(config["input"]+"/train-images-idx3-ubyte");
    std::string mnist_labels(config["input"]+"/train-labels-idx1-ubyte");


    return;

}
