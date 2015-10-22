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

#include <string>
#include <vector>
#include <stdint.h>

struct recognition_state_t {
    std::string   label;     // description of class
    std::string   wwid;      // wwid of recognised class
    float         accuracy;  // accuracy of recognition
};

struct image_recognition_item_t {
    std::string                       recognized_image;      // full path to recognized image
    std::string                       wwid;                  // original wwid of analysed image
    std::vector<recognition_state_t>  recognitions;
    std::vector<float>                nnet_output;
};

struct images_recognition_batch_t {
    uint64_t                               time_of_recognizing;
    uint64_t                               clocks_of_recognizing;
    std::vector<image_recognition_item_t>  recognized_images;    //

};
class C_report_maker
{
private:
    uint32_t batch_size;
    uint32_t execute_loops;
    std::string model;
    std::string device_name;
    std::string appname;
public:
    std::vector <images_recognition_batch_t> recognized_batches;  // List of recognized images, included information about performance also

public:
    C_report_maker(std::string _appname,
                   std::string _device_name,
                   std::string _model,
                      uint32_t _batch_size);
    ~C_report_maker();

public:
    bool print_to_html_file(std::string filename, std::string _title);
    bool print_to_csv_file(std::string filename);
    bool print_output_to_cvs_file(std::string filename);

};

