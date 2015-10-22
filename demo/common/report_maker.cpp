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

#include "report_maker.h"
#include "common/time_control.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>


C_report_maker::C_report_maker(std::string _appname,
                               std::string _device_name,
                               std::string _model,
                                  uint32_t _batch_size)
{
    appname = _appname;
    device_name = _device_name;
    model = _model;
    batch_size = _batch_size;
}


C_report_maker::~C_report_maker()
{
    for(auto itr1=recognized_batches.begin(); itr1!=recognized_batches.end(); ++itr1){
        for(auto itr2=itr1->recognized_images.begin(); itr2!=itr1->recognized_images.end();++itr2)
            itr2->recognitions.clear();
        itr1->recognized_images.clear();
    }
    recognized_batches.clear();
}

bool C_report_maker::print_to_html_file(std::string filename, std::string title)
{
    std::fstream html_file;
    bool result = true;
    uint32_t     batch_numerator = 0;
    html_file.open(filename, std::ios::out | std::ios::trunc);
    if(html_file.is_open())
    {
        // begin HTML file
        html_file <<
R"(<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>)"<< title<< R"(</title>
  <style>
     body { font-family: sans-serif;}
     p, h1, h2, th, td, li {padding: 0.5em;}
     img {margin: 0.5em 0px 0px 0.5em;}
     table.recognitions { padding:0.5em;}
     .thright { text-align:right;}
     .recognition {font-family:monospace;
       vertical-align:top;width:300px;
       -webkit-box-shadow: 5px 5px 9px 1px rgba(0,0,0,0.79);
       -moz-box-shadow: 5px 5px 9px 1px rgba(0, 0, 0, 0.79);
       box-shadow: 5px 5px 9px 1px rgba(0, 0, 0, 0.79);}
     .goal { font-weight: bold; color: red;}
     </style>
      </head>
      <body>)";
        // HTML file content
        {
            html_file << "<h2>" <<title<< "</h2>"<< std::endl;
            html_file << "<ul> "
                      << "<li> application: "<<appname
                      << "<li> device: "<<device_name
                      << "<li> model: " << model
                      << "<li> batch size: "<< batch_size
                      << "</ul><hr><hr>"<< std::endl;

            auto rec_batch_itr = recognized_batches.begin(),
                rec_batch_end = recognized_batches.end();
            for(;rec_batch_itr!=rec_batch_end;++rec_batch_itr)  {  // iterating batches
                //  html_file << "" << std::endl;
                html_file << "<table border=\"1\"><tr><th><h3> Batch ("<< ++batch_numerator<<")</h3></th><th>time</th><th>ticks</th></tr>" << std::endl;
                html_file << "<tr><th class=\"thright\">batch</th><td>"
                    << C_time_control::time_diff_string(rec_batch_itr->time_of_recognizing)
                    <<"</td><td>"
                    << C_time_control::clocks_diff_string(rec_batch_itr->clocks_of_recognizing)
                    <<"</td></tr>" << std::endl;

                html_file << "<tr><th class=\"thright\">per single image</th><td>"
                    << C_time_control::time_diff_string(rec_batch_itr->time_of_recognizing/batch_size)
                    <<"</td><td>"
                    << C_time_control::clocks_diff_string(rec_batch_itr->clocks_of_recognizing/batch_size)
                    <<"</td></tr></table>" << std::endl;

                html_file << "<table class=\"recognitions\" width=\""<<301*rec_batch_itr->recognized_images.size()<<"px\"><tr>" << std::endl;

                auto rec_img_itr = rec_batch_itr->recognized_images.begin(),
                    rec_img_end = rec_batch_itr->recognized_images.end();
                for(;rec_img_itr!=rec_img_end;++rec_img_itr) {    // iterating images within batch
                    html_file << "<td class=\"recognition\"><img height=\"150\" alt=\""
                        << rec_img_itr->recognized_image
                        <<"\" src=\""
                        << rec_img_itr->recognized_image
                        << "\">" << std::endl;
                    html_file << "<p> Original wwid: <b>"
                        << rec_img_itr->wwid
                        << "</b></p><p>Recognitions:</p><ol>" << std::endl;

                    auto rec_list_itr = rec_img_itr->recognitions.begin(),
                        rec_list_end = rec_img_itr->recognitions.end();
                    for(;rec_list_itr!=rec_list_end;++rec_list_itr) {
                        std::ostringstream rounded_float;
                        rounded_float.str("");
                        rounded_float << std::setprecision(1) << std::fixed << rec_list_itr->accuracy*100<<"% ";

                        html_file << "<li>"
                            << rounded_float.str()
                            << "[";
                        if(rec_img_itr->wwid.compare(rec_list_itr->wwid)==0)
                            html_file <<"<span style=\"font-weight: bold; color: red;\">"<< rec_list_itr->wwid<<"</span>";
                        else
                           html_file << rec_list_itr->wwid;
                        html_file  << "] "
                            << rec_list_itr->label
                            << "</li>"
                            << std::endl;
                    }
                    html_file << "</ol>" << std::endl;
                    html_file <<"    </td>";

                }
                html_file << "</tr></table><hr>" << std::endl;
            }
        }
        // ending HTML file
        html_file <<std::endl<<" </body>"<<std::endl<<"</html>"
            << std::endl;
        html_file.close();

    }
    else
    {
        std::cerr << "file access denied" << std::endl;
        result = false;
    }

    return result;
}

bool C_report_maker::print_to_csv_file(std::string filename){
  std::fstream csv_file;
  bool         result = true;
  uint32_t     batch_numerator = 0;
  csv_file.open(filename, std::ios::out | std::ios::trunc);
  if (csv_file.is_open())
  {
    auto rec_batch_itr = recognized_batches.begin(),
      rec_batch_end = recognized_batches.end();
    for (; rec_batch_itr != rec_batch_end; ++rec_batch_itr){
      auto rec_img_itr = rec_batch_itr->recognized_images.begin(),
        rec_img_end = rec_batch_itr->recognized_images.end();
      for (; rec_img_itr != rec_img_end; ++rec_img_itr)
      {
        auto reco_pos = 1;
        for (auto element : rec_img_itr->recognitions)
        {
          if (element.wwid.compare(rec_img_itr->wwid) == 0) break;
          reco_pos++;
        }
        csv_file
          << rec_img_itr->recognized_image << ";"
          << rec_batch_itr->time_of_recognizing / this->batch_size << ";"
          << rec_img_itr->wwid << ";"
          << reco_pos << ";"
          << std::endl;
      }
    }
  }
  csv_file.close();
  return result;
}
bool C_report_maker::print_output_to_cvs_file(std::string filename){
    std::fstream csv_file;
    bool         result = true;
    csv_file.open(filename, std::ios::out | std::ios::trunc);
    auto rec_batch_itr = recognized_batches.begin(),
        rec_batch_end = recognized_batches.end();
    for(; rec_batch_itr != rec_batch_end; ++rec_batch_itr){
        auto rec_img_itr = rec_batch_itr->recognized_images.begin(),
            rec_img_end = rec_batch_itr->recognized_images.end();
        for(; rec_img_itr != rec_img_end; ++rec_img_itr) {
            csv_file << rec_img_itr->recognized_image << ";";
            auto  nnet_output_itr = rec_img_itr->nnet_output.begin(),
                nnet_output_itr_end = rec_img_itr->nnet_output.end();
            for(; nnet_output_itr != nnet_output_itr_end; ++nnet_output_itr)
                csv_file << *nnet_output_itr <<";";
            csv_file << std::endl;
        }
    }
    csv_file.close();
    return result;
}
