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
#include "common/common_tools.h"

#include <iomanip>
#include <memory>
#include <cstring>

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
    uint32_t       loops = std::stoi(config["loops"]);
    bool           save_reference = (config.find("save_reference") != std::end(config));

    // load images from input directory
    auto images_list = get_directory_images(config["input"]);

    if(images_list.empty()) {
        throw std::runtime_error(std::string("error: directory ")+config["input"]+" does not contain any images that can be processed");
    }
    // 1000 classes as a workload output
    auto              workload_output       = new nn::data< float, 2 >( 1000, config_batch );
    if(workload_output == nullptr)   throw std::runtime_error("unable to create workload_output for batch = " +std::to_string(config_batch));

    nn::data< float > *output_array_cmpl[1] = { nn::data_cast< float, 0 >( workload_output ) };

    const std::string path( argv[0] );
    std::string appname( path.substr( path.find_last_of( "/\\" ) + 1 ) );

    C_report_maker report( appname, library.name, config["model"], config_batch );

    std::cout << "recognizing " << images_list.size() << " image(s)" << std::endl;
    std::cout << "loops: " << loops << std::endl;

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

        if (strcmp(config["model"].c_str(), "googlenet_float"))
        {
            images = nn_data_load_from_image_list(&batch_images,
                builder->get_img_size(), builder->image_process, config_batch,
                builder->RGB_order);
        }
        else
        {
            images = nn_data_load_from_image_list_with_padding(&batch_images,
                builder->get_img_size(), builder->image_process, config_batch,
                3, // TODO: remove hardcoded values
                2,
                3,
                2,
                builder->RGB_order
                );
        }


        if( images )
        {
            images_recognition_batch_t temp_report_recognition_batch;
            nn_data_t                  *input_array[1] = {images};
            {
                NN_API_STATUS  status;
                C_time_control timer;
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

            for( auto &image_filename : batch_images) {

                image_recognition_item_t temp_report_recognition_item;

                recognition_state_t     temp_recognition_state;
                std::map < float, int > output_values;

                temp_report_recognition_item.recognitions.clear();
                temp_report_recognition_item.recognized_image = image_filename;

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
                if(save_reference) {
                    std::string reference_output_filename = image_filename + ".txt";
                    std::fstream reference_output_file;
                    reference_output_file.open(reference_output_filename,std::ios::out | std::ios::trunc);

                    if(reference_output_file.is_open()) {
                        for(int index = 0; index < 1000; ++index)
                            reference_output_file << std::fixed << std::setprecision(8) << value_cmpl[index] << std::endl;
                        reference_output_file.close();
                    }
                    else {
                        std::cerr << "error: access denied - file: " << reference_output_filename << std::endl;
                    }
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
    unsigned int total_num_images_to_classify = static_cast<unsigned int>((images->size[2] / config_batch ) * config_batch + (rem_images != 0)*config_batch);
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

            for(int batch = num_images_to_ignore; batch < config_batch; ++batch )
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

void run_images_training( scoped_library      &library,
                         scoped_device       &device,
                         scoped_interface_0  &interface_0,
                         nn_workload_t *workload,
                         workflow_builder_base* builder,
                         char *argv[],
                         std::map<std::string, std::string> &config,
                         const int config_batch )
{
    std::cout << "--> PREPARING TRAINING DATA" << std::endl;

    std::srand ( 0x100 );

    auto imagedir = config["input"];

    // Load whole data into nn_datas.
    auto training_data = get_directory_train_images_and_labels(imagedir, builder->image_process, builder->get_img_size(), config_batch, builder->RGB_order);
    auto val_data = get_directory_val_images_and_labels(imagedir, builder->image_process, builder->get_img_size(), config_batch, builder->RGB_order);

    std::vector<int> indices(training_data.images->size[3]);
    for (uint32_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    auto dropout_seed = new nn::data<int32_t, 1>(1);
    auto execution_mode = new nn::data<int32_t, 1>(1);
    auto learning_rate = new nn::data<float, 1>(1);

    (*dropout_seed)(0) = 1;

    void* input_datas[5] =
    {
        nullptr,
        nullptr,
        dropout_seed,
        execution_mode,
        learning_rate
    };

    auto output = new nn::data<float, 2>(1000, config_batch);
    auto output_loss = new nn::data<float, 1>(1);

    void* output_datas[2] =
    {
        output,
        output_loss
    };

    auto loops = static_cast<uint32_t>(std::stoi( config["loops"] ));
    auto num_mini_batches = training_data.images->size[3] / config_batch;
    auto num_validation_packages = val_data.images->size[3] / config_batch;

    auto image_size = training_data.images->size[0] * training_data.images->size[1] *training_data.images->size[2];
    auto image_package_size = image_size * config_batch;
    auto label_size = training_data.labels->size[0];
    auto label_package_size = label_size * config_batch;

    // Temporary buffer later used for shuffling.
    std::vector<float> temporary_buffer(image_size);

    std::cout << "--> TRAINING STARTED with " << num_mini_batches << " minibatches of size " << config_batch << std::endl;

    // Initial.
    float rate = 0.01f;
    NN_API_STATUS status;
    float least_error = 0.0f;
    for(uint32_t epoch = 1; epoch <= loops; ++epoch)
    {
        std::cout << "Starting epoch " << epoch << std::endl;

        // Shuffling code.
        {
            // Shuffle indices.
            std::random_shuffle(indices.begin(), indices.end());

            // Save first elements for shuffling.
            memcpy(temporary_buffer.data(), training_data.images->buffer, image_size*sizeof(float));
            float temp_label = static_cast<float>((*training_data.labels)(0, 0));

            // Shuffle data according to indices.
            uint32_t previous_index = 0;
            for (uint32_t i = 0; i < indices.size(); ++i)
            {
                memcpy(
                    static_cast<float*>(training_data.images->buffer) + previous_index*image_size,
                    static_cast<float*>(training_data.images->buffer) + indices[i]*image_size,
                    image_size*sizeof(float));

                (*training_data.labels)(0, previous_index) = (*training_data.labels)(0, indices[i]);

                previous_index = indices[i];
            }

            // Dump saved elements to their new place.
            memcpy(static_cast<float*>(training_data.images->buffer) + previous_index*image_size, temporary_buffer.data(), image_size*sizeof(float));
            (*training_data.labels)(0, previous_index) = static_cast<int32_t>(temp_label);
        }

        // Update learning rate.
        if(epoch % 50 == 0)
            rate *= 0.5f;

        (*learning_rate)(0) = rate;
        (*execution_mode)(0) = 1; // Training mode.

        // Run all training images randomly.
        for(uint32_t mini_batch = 0; mini_batch < num_mini_batches; ++mini_batch)
        {
            // Increment dropout seed.
            (*dropout_seed)(0) += 1;

            std::cout << "." << std::flush;

            nn::data<float, 4> input_view(static_cast<float*>(training_data.images->buffer) + mini_batch*image_package_size, 3, builder->get_img_size(), builder->get_img_size(), config_batch);
            nn::data<int32_t, 2> label_view(static_cast<int32_t*>(training_data.labels->buffer) + mini_batch*label_package_size, 1, config_batch);

            input_datas[0] = &input_view;
            input_datas[1] = &label_view;

            status = interface_0.workload_execute_function(workload, input_datas, output_datas, &status);
            if(status != NN_API_STATUS_OK)
                throw std::runtime_error("api_status returned error");
        }

        float total_error = 0.0f;
        (*execution_mode)(0) = 0; // Validation mode.

        // Validate on validation set.
        for(uint32_t package = 0; package < num_validation_packages; ++package)
        {
            std::cout << ":" << std::flush;

            nn::data<float, 4> input_view(static_cast<float*>(val_data.images->buffer) + package*image_package_size, 3, builder->get_img_size(), builder->get_img_size(), config_batch);
            nn::data<int32_t, 2> label_view(static_cast<int32_t*>(val_data.labels->buffer) + package*label_package_size, 1, config_batch);

            input_datas[0] = &input_view;
            input_datas[1] = &label_view;

            status = interface_0.workload_execute_function(workload, input_datas, output_datas, &status);
            if(status != NN_API_STATUS_OK)
                throw std::runtime_error("api_status returned error");

            total_error += (*output_loss)(0);
        }

        std::cout << "\tError: " << total_error << std::endl;

        if(epoch == 1)
            least_error = total_error;
        else if((epoch==100) && total_error < least_error)
        {   // We've got better validation results than previously, dump weights and save new best result.
            least_error = total_error;

            std::cout << "Dumping params";
            uint32_t num_params;
            nn_workload_params* params = nullptr;
            status = interface_0.workload_query_param_function(workload, &params, &num_params);
            if(status != NN_API_STATUS_OK)
                throw std::runtime_error("api_status returned error");

            for(uint32_t param = 0; param < num_params; ++param)
            {
                nn::data<float> *returned_param = nullptr;
                if(params[param].dimensions == 4)
                    returned_param = new nn::data<float>(params[param].sizes[0], params[param].sizes[1], params[param].sizes[2], params[param].sizes[3]);
                else if(params[param].dimensions == 2)
                    returned_param = new nn::data<float>(params[param].sizes[0], params[param].sizes[1]);
                else if(params[param].dimensions == 1)
                    returned_param = new nn::data<float>(params[param].sizes[0]);
                else
                    throw std::invalid_argument("weight_save: wrong dimensionality returned");

                status = interface_0.workload_recover_param_function(workload, const_cast<char*>(params[param].name), returned_param);
                if(status != NN_API_STATUS_OK)
                {
                    delete returned_param;
                    throw std::runtime_error("api_status returned error");
                }

                std::cout << "." << std::flush;
                nn_data_save_to_file(returned_param, std::string("weights_caffenet_training/") + params[param].name + ".nnd");
                std::cout << ":" << std::flush;

                delete returned_param;
            }

            std::cout << std::endl;
            std::cout << "Dumping completed" << std::endl;
        }
    }

    interface_0.workload_delete_function( workload );

    return;
}
