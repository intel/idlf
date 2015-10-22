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

#pragma once
#include "device/api/nn_data_0.h"
#include "device/api/nn_primitives_api_0.h"
#include "common/FreeImage64/FreeImage.h"
#include "common/FreeImage_wraps.h"
#include <iostream>
#include <string>
#include <vector>

typedef struct nn_training_descriptor_s
{
    nn::data<float, 4>* images;
    nn::data<int32_t, 2>* labels;
} nn_training_descriptor_t;

#pragma pack(push,1)   /* The data has been redefined (alignment 4), so the pragma pack is not necessary,
                          but who knows if in the future, the compiler does not align to 8?  */
typedef struct nn_data_file_head_s {
    // Data header                   // Size [B]
    uint8_t    magic[3];             // 3          |
    uint8_t    data_type;            // 1           } aligment 2x4B
    uint8_t    version;              // 1          |
    uint8_t    dimension;            // 1
    uint8_t    sizeof_value;         // 1
} nn_data_file_head_t;

#pragma pack(pop)

#undef DC

uint32_t crc32( const void *buffer,size_t count,uint32_t crc_ini );

bool nn_data_save_to_file( nn::data<float>* in, std::string filename );

nn::data<float>* nn_data_convert_weights_2D_to_4D(nn::data<float>*src, uint32_t size_x,uint32_t size_y, uint32_t size_z, uint32_t size_n);

nn::data<float>* nn_data_load_from_file( std::string filename );

nn::data<float> *nn_data_load_from_file_time_measure(std::string filename,std::string note="");

nn::data<float, 3>* nn_data_load_from_image(std::string filename,
                                            uint32_t  std_size,
                                            fi::prepare_image_t image_process,
                                            bool RGB_order=true);

nn::data<float, 4>* nn_data_load_from_image_list(std::vector<std::string> *filelist,
                                                 uint32_t std_size,
                                                 fi::prepare_image_t image_process,
                                                 uint32_t batching_size,
                                                 bool RGB_order=true);

nn::data<float, 4>* nn_data_load_from_image_list_with_padding(std::vector<std::string> *filelist,
                                                 uint16_t std_size,
                                                 fi::prepare_image_t image_process,
                                                 uint16_t batching_size,
                                                 const size_t output_padding_left,
                                                 const size_t output_padding_right,
                                                 const size_t output_padding_top,
                                                 const size_t output_padding_bottom,
                                                 bool RGB_order = true);

nn_training_descriptor_t get_directory_train_images_and_labels(
                                                std::string images_path,
                                                fi::prepare_image_t image_process,
                                                uint32_t std_size,
                                                uint32_t batch,
                                                bool RGB_order);

nn_training_descriptor_t get_directory_val_images_and_labels(
                                                std::string images_path,
                                                fi::prepare_image_t image_process,
                                                uint32_t std_size,
                                                uint32_t batch,
                                                bool RGB_order);

nn::data<int32_t, 2>*  nn_data_load_from_label_list(std::vector<int32_t>*  labels,
                                                  uint32_t batching_size);

nn::data<float>* nn_data_extend_weights_by_padding( nn::data<float>* weights, uint32_t extended_num_input_feature_maps, uint32_t extended_num_output_feature_maps);
nn::data<float>* nn_data_extend_biases_by_padding( nn::data<float>* biases, uint32_t required_num_outputs);

void nn_data_convert_float_to_int16_fixedpoint(nn::data<float> *in,
                                               nn::data<int16_t> *out,
                                               float scale);

void nn_data_convert_float_to_int32_fixedpoint(nn::data<float> *in,
                                               nn::data<int32_t> *out,
                                               float scale);

nn::data<float, 4> *nn_data_load_from_image_list_for_int16(std::vector<std::string> *filelist,
                                                           uint32_t std_size,
                                                           uint32_t batching_size);
void nn_data_load_images_and_labels_from_mnist_files( nn::data< float, 3 >* &images,
                                                      nn::data< char, 1 >* &labels,
                                                      std::string &mnist_images,
                                                      std::string &mnist_labels );
