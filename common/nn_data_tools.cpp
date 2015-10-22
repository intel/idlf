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

#include "common/nn_data_tools.h"
#include "common/time_control.h"
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
                                             uint32_t std_size,     // size of image both: height and width
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
        auto data_buffer = static_cast<float*>(data->buffer);
        if(RGB_order) {
            for(uint32_t y=0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for(uint32_t x=0u; x<std_size; ++x) {
                    *(data_buffer + 0 + x*3 + y*3*std_size) = pixel[FI_RGBA_RED];
                    *(data_buffer + 1 + x*3 + y*3*std_size) = pixel[FI_RGBA_GREEN];
                    *(data_buffer + 2 + x*3 + y*3*std_size) = pixel[FI_RGBA_BLUE];
                    pixel += bytes_per_pixel;
                }
            }
        }
        else {
            for(uint32_t y=0u; y<std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for(uint32_t x=0u; x<std_size; ++x) {
                    *(data_buffer + 0 + x*3 + y*3*std_size) = pixel[FI_RGBA_BLUE];
                    *(data_buffer + 1 + x*3 + y*3*std_size) = pixel[FI_RGBA_GREEN];
                    *(data_buffer + 2 + x*3 + y*3*std_size) = pixel[FI_RGBA_RED];
                    pixel += bytes_per_pixel;
                }
            }
        }
        FreeImage_Unload(bitmap);
        return data;
    }
    return nullptr;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float>* nn_data_extend_biases_by_padding( nn::data<float>* biases, uint32_t required_num_outputs)
{
    // Check if number of out FM (dim number 1)  is smaller than required (to be extended to) number of dimensions
    if (biases->size[0] >= required_num_outputs)
    {
        assert(0);
        return nullptr;
    }

    // Make a new biases container with dimension corressponding to number of outputs (num filters)
    size_t sizes[1] = {required_num_outputs};
    nn::data<float>* extended_biases = new nn::data<float>(sizes,1);

    for(uint32_t n = 0; n< extended_biases->size[0]; ++n)
    {
        if(n < biases->size[0])
        {
            extended_biases->at(n) = biases->at(n);
        } else {
            extended_biases->at(n) = 0.0f;
        }
    }

    return extended_biases;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float>* nn_data_extend_weights_by_padding( nn::data<float>* weights, uint32_t extended_num_input_feature_maps, uint32_t extended_num_output_feature_maps)
{
    // Check if number of out FM (dim number 3)  is smaller than required (to be extended to) number of dimensions
    // TODO: Make it better validation of sizes
    if(weights->size[3] > extended_num_output_feature_maps )
    {
        assert(0);
        return nullptr;
    }

    if(weights->size[2] > extended_num_input_feature_maps )
    {
        assert(0);
        return nullptr;
    }

    // Make a new weights container with dimension corressponding to number of outputs (num filters)
    size_t sizes[4] = {weights->size[0],weights->size[1],extended_num_input_feature_maps, extended_num_output_feature_maps};
    nn::data<float>* extended_weights = new nn::data<float>(sizes,4);

    // Copy source weights into extended container and fill the remaining part woth zeros
    for(uint32_t n = 0; n< extended_weights->size[3]; ++n)
    {
        if(n < weights->size[3])
        {
            for(uint32_t z = 0; z< extended_weights->size[2]; ++z)
            {
                if(z < weights->size[2])
                {
                    for(uint32_t y = 0; y< weights->size[1]; ++y)
                    {
                        for(uint32_t x = 0; x< weights->size[0]; ++x)
                        {
                            extended_weights->at(x,y,z,n) = weights->at(x,y,z,n);
                        }
                    }
                } else {
                    for(uint32_t y = 0; y< extended_weights->size[1]; ++y)
                    {
                        for(uint32_t x = 0; x< extended_weights->size[0]; ++x)
                        {
                             extended_weights->at(x,y,z,n) = 0.0f;
                        }
                    }
                }
            }
        } else {
            for(uint32_t z = 0; z< extended_weights->size[2]; ++z)
            {
                for(uint32_t y = 0; y< weights->size[1]; ++y)
                {
                    for(uint32_t x = 0; x< weights->size[0]; ++x)
                    {
                         extended_weights->at(x,y,z,n) = 0.0f;
                    }
                }
            }
        }
    }

    return extended_weights;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float>* nn_data_convert_weights_2D_to_4D(nn::data<float>*src, uint32_t size_x,uint32_t size_y, uint32_t size_z, uint32_t size_n)
{
    // TODO: make it more generic eg. size 1D upto 3D can be converted into 4D
    if(src->dimension != 2)
    {
        assert(0);
        throw std::runtime_error("conversion of fully connected weights did not succeed!");
        return nullptr;
    }

    size_t sizes[4] = {size_x,size_y,size_z,size_n};
    nn::data<float>* converted_weights = new nn::data<float>(sizes,4);

    // Out of first dimension of source nn::data we will make content to first three dimensions of destination container
    for(uint32_t n = 0; n< size_n; ++n)
    {
        size_t idx = 0;
        for(uint32_t z = 0; z< size_z; ++z)
        {
            for(uint32_t y = 0; y< size_y; ++y)
            {
                for(uint32_t x = 0; x< size_x; ++x)
                {
                    converted_weights->at(x,y,z,n) = src->at(idx++,n);
                }
            }
        }
    }

    return converted_weights;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float, 4>*  nn_data_load_from_image_list(          // Load of all data from a batch of image files
    std::vector<std::string>*  filelist,                    // Pointer to vector contained a batch of filenames of images
    uint32_t  std_size,                                     // All images will be converted to uniform size -> std_size
    fi::prepare_image_t image_process,                      // Pointer of function for image processing
    uint32_t batching_size,                                 // A portion of the images must have a specific size
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
        nn::data<float, 3> view(buffer + index*image_stride, 3, std_size, std_size);
        if(it!=filelist->end()) {
            auto image = nn_data_load_from_image(*it, std_size, image_process, RGB_order);
            if(image) memcpy(view.buffer, image->buffer, view.count()*sizeof(float));
            else set_zero(view);
            delete image;
            ++it;
        }
        else set_zero(view);
    }
    return result;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float, 4>*  nn_data_load_from_image_list_with_padding(     // Load of all data from a batch of image files
    std::vector<std::string>*  filelist,                            // Pointer to vector contained a batch of filenames of images
    uint16_t  std_size,                                             // All images will be converted to uniform size -> std_size
    fi::prepare_image_t image_process,                              // Pointer of function for image processing
    uint16_t batching_size,                                         // A portion of the images must have a specific size
    const size_t output_padding_left,
    const size_t output_padding_right,
    const size_t output_padding_top,
    const size_t output_padding_bottom,
    bool RGB_order)                                                 // If true, then images are load with RGB order, otherwise BGR
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
    auto image_stride = 3 * std_size*std_size;
    for (auto index = 0u; index < batching_size; ++index) {
        float *buffer = reinterpret_cast<float *>(result->buffer);
        nn::data<float, 3> view(buffer + index*image_stride, 3, std_size, std_size);
        set_zero(view);

        if (it != filelist->end()) {
            size_t img_size = 256;// std_size - (output_padding_left + output_padding_right);
            size_t dest_size = 224;
            auto image = nn_data_load_from_image(*it, img_size, image_process, RGB_order);
            if (image) {
                for (int y = 0; y < dest_size; ++y) {
                    memcpy((float *)view.buffer + (output_padding_left + (output_padding_top + y) * std_size) * 3, (float *)image->buffer + (((y + (img_size - dest_size) / 2)*img_size + (img_size - dest_size) / 2)) * 3, 3 * dest_size*sizeof(float));
                    //memcpy(&view.at(0, output_padding_left, output_padding_top + y), &image->at(0, 0, y), 3 * img_size * sizeof(float));
                }
            }
            else set_zero(view);
            delete image;
            ++it;
        }
        else set_zero(view);
    }

    return result;

}

nn::data<int32_t, 2>*  nn_data_load_from_label_list(
    std::vector<int32_t>*  label_list,
    uint32_t batching_size)
    // supported formats: TXT
{
    auto result = new nn::data<int32_t, 2>(1, batching_size);
    uint32_t index = 0;
    for(auto it = label_list->begin(); it != label_list->end(); ++it, ++index)
        (*result)(0,index) = *it;

    return result;
}


nn_training_descriptor_t get_directory_train_images_and_labels(
    std::string images_path,
    fi::prepare_image_t image_process,
    uint32_t std_size,
    uint32_t batch,
    bool RGB_order)
{
    std::ifstream lifs( images_path+"/train.txt" );

    std::vector<std::string> image_names;
    std::vector<int32_t> label_ids;

    while( !lifs.eof() )
    {
        std::string image_name;
        int32_t label_id;
        lifs >> image_name >> label_id;

        image_names.push_back(images_path + "/" + image_name);
        label_ids.push_back(label_id);
    }

    auto elements = image_names.size();
    std::cout << "Found " << elements << " training images descriptors" << std::endl;

    if(elements % batch != 0)
        throw std::invalid_argument("get_train_images: number of descriptors not adjusted to batch size");

    if(image_names.size() != label_ids.size())
        throw std::invalid_argument("get_train_images: number of labels is different from number of images as described in training list");

    nn_training_descriptor_t result = {nn_data_load_from_image_list(
            &image_names,
            static_cast<uint32_t>(std_size),
            image_process,
            static_cast<uint32_t>(elements),
            RGB_order),
            nn_data_load_from_label_list(&label_ids, static_cast<uint32_t>(elements))
    };

    if(result.images->size[3] != result.labels->size[1])
        throw std::invalid_argument("get_train_images: loaded number of labels is different from loaded number of images");

    if(result.images->size[3] != image_names.size())
        throw std::invalid_argument("get_train_images: number of loaded items is different than number of descriptors");

    std::cout << "Loaded " << result.images->size[3] << " training images" << std::endl;
    return result;
}

nn_training_descriptor_t get_directory_val_images_and_labels(
    std::string images_path,
    fi::prepare_image_t image_process,
    uint32_t std_size,
    uint32_t batch,
    bool RGB_order)
{
    std::ifstream lifs( images_path+"/val.txt" );

    std::vector<std::string> image_names;
    std::vector<int32_t> label_ids;

    while( !lifs.eof() )
    {
        std::string image_name;
        int32_t label_id;
        lifs >> image_name >> label_id;

        image_names.push_back(images_path + "/" + image_name);
        label_ids.push_back(label_id);
    }

    auto elements = image_names.size();
    std::cout << "Found " << elements << " validation images descriptors" << std::endl;

    if(elements % batch != 0)
        throw std::invalid_argument("get_val_images: number of descriptors not adjusted to batch size");

    if(image_names.size() != label_ids.size())
        throw std::invalid_argument("get_val_images: number of labels is different from number of images as described in validation list");

    nn_training_descriptor_t result = {nn_data_load_from_image_list(
            &image_names,
            static_cast<uint32_t>(std_size),
            image_process,
            static_cast<uint32_t>(elements),
            RGB_order),
            nn_data_load_from_label_list(&label_ids, static_cast<uint32_t>(elements))
    };

    if(result.images->size[3] != result.labels->size[1])
        throw std::invalid_argument("get_val_images: loaded number of labels is different from loaded number of images");

    if(result.images->size[3] != image_names.size())
        throw std::invalid_argument("get_val_images: number of loaded items is different than number of descriptors");

    std::cout << "Loaded " << result.images->size[3] << " validation images" << std::endl;

    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Read all labels from given mnist labels file and all corressponding images from mnist images file
void nn_data_load_images_and_labels_from_mnist_files( nn::data< float, 3 >* &images, nn::data< char, 1 >* &labels, std::string &mnist_images, std::string &mnist_labels)
{
    char int_read[sizeof(uint32_t)];
    const int mnist_labels_magic_number = 2049;
    const int mnist_images_magic_number = 2051;

    // Convert 4 chars in big endian order into 4bytes long integer
    auto block_to_int = [](char *memblock)
    {
        uint32_t int_value = ((((((uint8_t)memblock[0] << 8) + (uint8_t)memblock[1]) << 8 )+ (uint8_t)memblock[2]) << 8)  + (uint8_t)memblock[3];
        return int_value;
    };

    //1. Read MNIST labels
    std::ifstream lifs;
    lifs.open(mnist_labels, std::ifstream::binary );

    if(lifs.good() == false) {
        std::string err_msg("Error reading: " + mnist_labels);
        throw std::runtime_error(err_msg.c_str());
    }

    // 1.1 Get Magic number
    lifs.read(int_read,sizeof(uint32_t));
    uint32_t magic_number = block_to_int(int_read);
    if( magic_number != mnist_labels_magic_number ) {
        std::string err_msg("Error reading MNIST Labels eg. Wrong Magic number read: ");
        err_msg += std::to_string(magic_number );
        throw std::runtime_error(err_msg);
    }

    // 1.2 Get Number of labels
    lifs.read(int_read,sizeof(uint32_t));
    uint32_t num_labels = block_to_int(int_read);

    // 1.3 Get labels
    labels = new nn::data<char, 1>(num_labels);
    char* pbuf = (char*)labels->buffer;
    for(unsigned int l=0 ; l< num_labels; ++l)
    {
        lifs.read(pbuf+l,sizeof(char));
    }

    lifs.close();

    //2. Read MNIST images
    std::ifstream ifs;
    ifs.open(mnist_images, std::ifstream::binary );

    if(ifs.good() == false) {
        std::string err_msg("Error reading: " + mnist_images);
        throw std::runtime_error(err_msg.c_str());
    }

    // 2.1 Get Magic number and number of images and width and height of each image
    ifs.read(int_read,sizeof(uint32_t));
    magic_number = block_to_int(int_read);
    if( magic_number != mnist_images_magic_number ) {
        std::string err_msg("Error reading MNIST Images eg. Wrong Magic number read: ");
        err_msg += std::to_string(magic_number);
        throw std::runtime_error(err_msg);
    }

    // Get Number of images
    ifs.read(int_read,sizeof(uint32_t));
    uint32_t num_images = block_to_int(int_read);

    // Get images' width
    ifs.read(int_read,sizeof(uint32_t));
    uint32_t images_width = block_to_int(int_read);

    // Get images' height
    ifs.read(int_read,sizeof(uint32_t));
    uint32_t images_height = block_to_int(int_read);

    // 2.2 Get all images
    images = new nn::data<float, 3>(images_width, images_height, num_images);
    std::unique_ptr<char[]> single_image(new char[images_height * images_width]);

    // Read all images (all at one) , scale its values from: <0-255> to <0.0 - 1.0>
    // as caffe training was processing data that way
    for(unsigned int img=0; img < num_images; ++img)
    {
        ifs.read(single_image.get(),images_width*images_height*sizeof(char));

        float* pdata_image = (float*)images->buffer + img*images_width*images_height;   //?? is it safe
        for(unsigned int pix=0; pix < images_width*images_height; ++pix )
        {
             pdata_image[pix]= *((unsigned char*)single_image.get() + pix)*1.0f/255.0f;
        }

    }

    ifs.close();

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void nn_data_convert_float_to_int16_fixedpoint(nn::data<float> *in,
                                                        nn::data<int16_t> *out,
                                                        const float scale){
    const size_t N = in->count();
    auto  inbuffer = static_cast<float *>(in->buffer);
    auto outbuffer = static_cast<int16_t *>(out->buffer);

    for (size_t i = 0; i < N; ++i)
        outbuffer[i] = static_cast<int16_t>(inbuffer[i] * scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void nn_data_convert_float_to_int32_fixedpoint(nn::data<float> *in,
                                                        nn::data<int32_t> *out,
                                                        float scale){
    const size_t N = in->count();
    auto inbuffer = static_cast<float *>(in->buffer);
    auto outbuffer = static_cast<int32_t *>(out->buffer);

    for(size_t i=0;i<N;++i)
        outbuffer[i] =  static_cast<int32_t>(inbuffer[i] * scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn::data<float, 4>*  nn_data_load_from_image_list_for_int16(    // Load of all data from a batch of image files
    std::vector<std::string>*  filelist,                                 // Pointer to vector contained a batch of filenames of images
    uint32_t  std_size,                                                  // All images will be converted to uniform size -> std_size
    uint32_t batching_size)                                              // A portion of the images must have a specific size
    // supported formats: JPEG, J2K, JP2, PNG, BMP, WEBP, GIF, TIFF
{
    auto data = new nn::data<float, 4>(3, std_size, std_size, batching_size);

    auto filelist_count = filelist->size();

    // Verifying that the images on the list is more than the size of batching
    uint32_t  count = (filelist_count > batching_size) ? batching_size : static_cast<uint32_t>(filelist_count);

    // Read images from filelist vector
    uint32_t i = 0;
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
            for (uint32_t y = 0; y < std_size; ++y) {
                uint8_t *pixel = FreeImage_GetScanLine(bitmap, std_size - y - 1);
                for (uint32_t x = 0; x < std_size; ++x) {
                    data->at(0, x, y, i) = pixel[FI_RGBA_RED];
                    data->at(1, x, y, i) = pixel[FI_RGBA_GREEN];
                    data->at(2, x, y, i) = pixel[FI_RGBA_BLUE];
                    pixel += Bpp;
                }
            }
            FreeImage_Unload(bitmap);
        }
        else {  // What, if reading bitmap failed? - Why not fill with zeros?
            for (uint32_t y = 0; y < std_size; ++y) {
                for (uint32_t x = 0; x < std_size; ++x) {
                    for(uint32_t p = 0; p < 3; ++p) {
                        data->at(p, x, y, i) = 0;
                    }
                }
            }
        }
    }
    // Fill with zero values, if number of  images are less than batching size
    for (; i < batching_size; ++i) {
        for (uint32_t y = 0; y < std_size; ++y) {
            for (uint32_t x = 0; x < std_size; ++x) {
              for (uint32_t p = 0; p < 3;++p)
                data->at(i, x, y, p) = 0;
            }
        }
    }
    return data;
}
