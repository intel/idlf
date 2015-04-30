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

#include "nn_data_tools.h"
#include "time_control.h"
#include <fstream>
#include <cstring>
#include <nmmintrin.h>
#include <algorithm>
#include <chrono>
#include <memory>

#define CRC_INIT 0xbaba7007

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t crc32( const void *buffer,size_t count,uint32_t crc_ini )
{
    uint32_t crc = crc_ini;
    const uint8_t *ptr = static_cast<const uint8_t *>(buffer);
    while ( count >= 4 ) {
        crc = _mm_crc32_u32( crc,*reinterpret_cast<const uint32_t *>(ptr) );
        ptr += 4;
        count -= 4;
    }
    while ( count-- ) crc = _mm_crc32_u32( crc,*(ptr++) );
    return crc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool nn_data_save_to_file(                     // Storage of all data into a binary file
    nn::data<float>      *in,
    std::string           filename )
{
    try {
        std::ofstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(filename, std::ios::out | std::ios::trunc | std::ios::binary);
        const nn_data_file_head_t file_head = {
            {'N', 'N', 'D'},    // magic[]
            'F',                // data_type
            1,                  // version
            in->dimension,      // dimension
            in->sizeof_value    // sizeof_value
        };

        auto write_crc = [&file](uint32_t crc) -> void {
            file.write(reinterpret_cast<char *>(&crc), sizeof(uint32_t));
        };

        // write header with 32-bit crc
        file.write(reinterpret_cast<const char *>(&file_head), sizeof(file_head));
        write_crc(crc32(&file_head, sizeof(file_head), CRC_INIT));

        // write size array with 32-bit crc
        const char *array = reinterpret_cast<const char *>(in->nn_data_t::size);
        const auto array_size = in->dimension*sizeof(*in->nn_data_t::size);
        file.write( array, array_size);
        write_crc(crc32(array, array_size, CRC_INIT));

        // write data with 32-bit crc
        size_t buffer_size = in->sizeof_value;
        for(auto index=0u; index<in->dimension; ++index) buffer_size *= in->size[index];
        file.write(static_cast<char *>(in->buffer), buffer_size);
        write_crc(crc32(in->buffer, buffer_size, CRC_INIT));
    }
    catch(...) {
        return false;
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float>*  nn_data_load_from_file( std::string  filename )     // Load of all data from a binary file (version 2)
// Warning! This is not a generic but only for float data
// TODO: More generic
{
    try {
        std::ifstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(filename, std::ios::in | std::ios::binary);

        auto read_crc = [&file]() -> uint32_t {
            uint32_t result;
            file.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
            return result;
        };

        // load header, verify 32-bit crc
        nn_data_file_head_t file_head;
        file.read(reinterpret_cast<char *>(&file_head), sizeof(file_head));
        if(read_crc()!=crc32(&file_head, sizeof(file_head), CRC_INIT)) throw std::runtime_error("nn_data_t header crc mismatch");
        if(   file_head.sizeof_value!=sizeof(float)
           || file_head.data_type!='F' ) throw std::runtime_error("nn_data_t has invalid type");

        // load size array, verify 32-bit crc
        auto array = std::unique_ptr<size_t>(new size_t[file_head.dimension]);
        auto array_size = file_head.dimension*sizeof(size_t);
        file.read(reinterpret_cast<char *>(array.get()), array_size);
        if(read_crc()!=crc32(array.get(), array_size, CRC_INIT)) throw std::runtime_error("nn_data_t size array crc mismatch");

        // create target nn::data & load data into it
        auto data = std::unique_ptr<nn::data<float>>(new nn::data<float>(array.get(), file_head.dimension));
        auto data_size = data.get()->count()*sizeof(float);
        file.read(static_cast<char *>(data.get()->buffer), data_size);
        if(read_crc()!=crc32(data.get()->buffer, data_size, CRC_INIT)) throw std::runtime_error("nn_data_t data crc mismatch");

        // return result
        auto result = data.get();
        data.release();
        return result;
    }
    catch(...) {
        return nullptr;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float> *nn_data_load_from_file_time_measure(
    std::string filename,
    std::string note
    )
{
    std::cout << "Load file  " << filename << " " << note;
    C_time_control timer;
    nn::data<float> *nn_temp= nn_data_load_from_file(filename);
    timer.tock();
    std::cout<< " at " << timer.time_diff_string() <<" ("<< timer.clocks_diff_string() <<")"  <<std::endl;

    return nn_temp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float, 3>*  nn_data_load_from_image(std::string  filename, // Load of all data from a image filename
                                             uint16_t std_size,     // size of image both: height and width
                                             fi::prepare_image_t image_process, // pointer of function for image processing
                                             bool RGB_order)        // if true - image have RGB order, otherwise BGR
// supported formats: JPEG, J2K, JP2, PNG, BMP, WEBP, GIF, TIFF
{

    auto data = new nn::data<float,3>(3, std_size, std_size);
    if(FIBITMAP *bitmap_raw = fi::load_image_from_file( filename )) {
        FIBITMAP *bitmap;

        if(FreeImage_GetBPP(bitmap_raw)!=24) {
            bitmap = FreeImage_ConvertTo24Bits(bitmap_raw);
            FreeImage_Unload(bitmap_raw);
        } else bitmap = bitmap_raw;

        bitmap = image_process(bitmap, std_size);

        auto bytes_per_pixel = FreeImage_GetLine( bitmap )/std_size;
        if(RGB_order) {
            for(uint32_t y=0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for(uint32_t x=0u; x<std_size; ++x) {
                    data->at(0, x, y) = pixel[FI_RGBA_RED];
                    data->at(1, x, y) = pixel[FI_RGBA_GREEN];
                    data->at(2, x, y) = pixel[FI_RGBA_BLUE];
                    pixel += bytes_per_pixel;
                }
            }
        }
        else {
            for(uint32_t y=0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for(uint32_t x=0u; x<std_size; ++x) {
                    data->at(0, x, y) = pixel[FI_RGBA_BLUE];
                    data->at(1, x, y) = pixel[FI_RGBA_GREEN];
                    data->at(2, x, y) = pixel[FI_RGBA_RED];
                    pixel += bytes_per_pixel;
                }
            }

        }

        FreeImage_Unload(bitmap);
        return data;
    }
    return nullptr;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float, 4>*  nn_data_load_from_image_list(          // Load of all data from a batch of image files
    std::vector<std::string>*  filelist,                    // Pointer to vector contained a batch of filenames of images
    uint16_t  std_size,                                     // All images will be converted to uniform size -> std_size
    fi::prepare_image_t image_process,                      // Pointer of function for image processing
    uint16_t batching_size,                                 // A portion of the images must have a specific size
    bool RGB_order)                                         // If true, then images are load with RGB order, otherwise BGR
    // supported formats: JPEG, J2K, JP2, PNG, BMP, WEBP, GIF, TIFF
{
    auto set_zero = [](nn::data<float, 3> &dst) {
        for(auto z=0u; z<dst.size[2]; ++z)
            for(auto y=0u; y<dst.size[1]; ++y)
                for(auto x=0u; x<dst.size[0]; ++x)
                    dst(x,y,z) = 0.0f;
    };

    auto result = new nn::data<float, 4>(3, std_size, std_size, batching_size);
    auto it = filelist->begin();
    auto image_stride = 3*std_size*std_size;
    for(auto index=0u; index<batching_size; ++index) {
        float *buffer = reinterpret_cast<float *>(result->buffer);
        nn::data<float,3> view(buffer+index*image_stride, 3, std_size, std_size);
        if(it!=filelist->end()) {
            auto image = nn_data_load_from_image(*it, std_size, image_process, RGB_order);
            if(image) memcpy(view.buffer, image->buffer, view.count()*sizeof(float));
            else set_zero(view);
            delete image;
            ++it;
        } else set_zero(view);
    }
    return result;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void nn_data_convert_float_to_int16_fixedpoint(nn::data<float> *in,
                                                        nn::data<int16_t> *out,
                                                        const float scale){
    const size_t N = in->count();
    auto  inbuffer = static_cast<float *>(in->buffer);
    auto outbuffer = static_cast<int16_t *>(out->buffer);

    for (size_t i = 0; i < N; ++i)
        outbuffer[i] = inbuffer[i] * scale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void nn_data_convert_float_to_int32_fixedpoint(nn::data<float> *in,
                                                        nn::data<int32_t> *out,
                                                        float scale){
    const size_t N = in->count();
    auto  inbuffer = static_cast<float *>(in->buffer);
    auto outbuffer = static_cast<int32_t *>(out->buffer);

    for(size_t i=0;i<N;++i)
        outbuffer[i] = inbuffer[i] * scale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float, 4>*  nn_data_load_from_image_list_for_int16(    // Load of all data from a batch of image files
    std::vector<std::string>*  filelist,                                 // Pointer to vector contained a batch of filenames of images
    uint16_t  std_size,                                                  // All images will be converted to uniform size -> std_size
    uint16_t batching_size)                                              // A portion of the images must have a specific size
    // supported formats: JPEG, J2K, JP2, PNG, BMP, WEBP, GIF, TIFF
{
    auto data = new nn::data<float, 4>(3, std_size, std_size, batching_size);

    auto filelist_count = filelist->size();

    // Verifying that the images on the list is more than the size of batching
    uint16_t  count = (filelist_count > batching_size) ? batching_size : filelist_count;

    // Read images from filelist vector
    uint16_t i = 0;
    std::vector<std::string>::iterator files_itr = filelist->begin();
    for (; i < count; ++i, ++files_itr) {
        FIBITMAP *bitmap = fi::load_image_from_file(files_itr->data());
        if (bitmap) {
            if (FreeImage_GetBPP(bitmap) != 24) {
                FIBITMAP *bmp_temp = FreeImage_ConvertTo24Bits(bitmap);
                FreeImage_Unload(bitmap);
                bitmap = bmp_temp;
            }
            bitmap = fi::crop_image_to_square_and_resize(bitmap, std_size);
            uint8_t Bpp = FreeImage_GetLine(bitmap) / std_size; // And now Bpp means Bytes per pixel
            for (int y = 0; y < std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for (int x = 0; x < std_size; ++x) {
                    data->at(0, x, y, i) = pixel[FI_RGBA_RED];
                    data->at(1, x, y, i) = pixel[FI_RGBA_GREEN];
                    data->at(2, x, y, i) = pixel[FI_RGBA_BLUE];
                    pixel += Bpp;
                }
            }
            FreeImage_Unload(bitmap);
        }
        else {  // What, if reading bitmap failed? - Why not fill with zeros?
            for (int y = 0; y < std_size; ++y) {
                for (int x = 0; x < std_size; ++x) {
                    for(int p = 0; p < 3; ++p) {
                        data->at(p, x, y, i) = 0;
                    }
                }
            }
        }
    }
    // Fill with zero values, if number of  images are less than batching size
    for (; i < batching_size; ++i) {
        for (int y = 0; y < std_size; ++y) {
            for (int x = 0; x < std_size; ++x) {
              for (int p = 0; p < 3;++p)
                data->at(i, x, y, p) = 0;
            }
        }
    }
    return data;
}
