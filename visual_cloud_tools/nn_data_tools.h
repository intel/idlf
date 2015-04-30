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
#include "nn_data_0.h"
#include "nn_primitives_api_0.h"
#include "FreeImage_wraps.h"
#include <iostream>
#include <string>
#include <vector>

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

nn::data<float>* nn_data_load_from_file( std::string filename );

nn::data<float> *nn_data_load_from_file_time_measure(std::string filename,std::string note="");

nn::data<float, 3>* nn_data_load_from_image(std::string filename, 
                                            uint16_t  std_size, 
                                            fi::prepare_image_t image_process, 
                                            bool RGB_order=true);

nn::data<float, 4>* nn_data_load_from_image_list(std::vector<std::string> *filelist,
                                                 uint16_t std_size,
                                                 fi::prepare_image_t image_process,
                                                 uint16_t batching_size,
                                                 bool RGB_order=true);


void nn_data_convert_float_to_int16_fixedpoint(nn::data<float> *in,
                                               nn::data<int16_t> *out,
                                               float scale);

void nn_data_convert_float_to_int32_fixedpoint(nn::data<float> *in,
                                               nn::data<int32_t> *out,
                                               float scale);

nn::data<float, 4> *nn_data_load_from_image_list_for_int16(std::vector<std::string> *filelist,
                                                           uint16_t std_size,
                                                           uint16_t batching_size);