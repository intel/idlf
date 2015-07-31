/*
Copyright (c) 2015, Intel Corporation

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
#include "device/api/nn_device_interface_0.h"
#include "device/common/nn_workload_data.h"
#include "device/gpu/core/layers_opencl.h"
#include <vector>

// Make it C++ not a mess

struct nn_cl_data_fragment
{
    void* ptr;
    uint64_t size;
};

struct nn_cl_data_parent
{
    nn_workload_data_coords_t       lengths;        /* Data structure size in each dimension */
    nn_workload_data_layout_t       layout;         /* Data structure layout */

    uint64_t                        buffer_size;    /* Size of used buffer space */
    uint64_t                        buffer_aligned_size;    /* Size of total buffer space */

    device_gpu::ocl_toolkit        *device_context;

    uint64_t                        buffer_mask;

    uint32_t                        reference_counter;

    std::vector<cl::Buffer*>        cl_buffer;
    std::vector<cl::Image2D*>       cl_image;

    std::vector<std::vector<cl::Buffer*>>        cl_subbuffer;
};

struct nn_cl_data
{
    nn_workload_data_coords_t       view_begin;     /* first point of view [min corner of hyper parallelogram] */
    nn_workload_data_coords_t       view_end;       /* last point of view [max corner of hyper parallelogram] */

    nn_cl_data_parent              *parent;

    nn_cl_data(
        device_gpu::ocl_toolkit* context,
        uint64_t mask,
        nn_workload_data_t* source,
        void* ptr = nullptr,
        bool make_image = false,
        uint_least32_t         image_width = 0,
        uint_least32_t         image_height = 0,
        bool fragment_buffer = false,
        std::vector<nn_cl_data_fragment>* bases_sizes = nullptr)
    {
        parent = new nn_cl_data_parent;

        parent->lengths = source->parent->lengths;
        parent->layout = source->parent->layout;
        parent->device_context = context;
        parent->buffer_mask = mask;

        parent->reference_counter = 1;

        const uint64_t align = 4 * 4096;

        parent->buffer_size = source->parent->buffer_size;
        parent->buffer_aligned_size = (source->parent->buffer_size + align - 1) / align * align;

        view_begin = source->view_begin;
        view_end = source->view_end;

        cl_int err;

        if (fragment_buffer)
        {
            for (auto base_size : *bases_sizes)
            {
                auto base = base_size.ptr;
                auto size = base_size.size;

                parent->cl_buffer.push_back(new cl::Buffer(
                    parent->device_context->get_context(),
                    parent->buffer_mask & ~CL_MEM_COPY_HOST_PTR,
                    (size + align - 1) / align * align,
                    nullptr,
                    &err));

                if (err != CL_SUCCESS)
                    THROW_ERROR(err, "Error creating nn_cl_data fragmented buffer.");

                if (parent->buffer_mask & CL_MEM_COPY_HOST_PTR)
                {

                    float* mapped_ptr = static_cast<float*>(
                        clEnqueueMapBuffer(
                        parent->device_context->get_command_queue()(),
                        (*parent->cl_buffer.back())(),
                        true,
                        CL_MEM_READ_WRITE,
                        0,
                        size,
                        0,
                        nullptr,
                        nullptr,
                        &err));

                    if (err != CL_SUCCESS)
                        THROW_ERROR(err, "Error in mapping buffer at nn_cl_data buffer creation with HOST COPY.");

                    memcpy(mapped_ptr, base, size);

                    clEnqueueUnmapMemObject(
                        parent->device_context->get_command_queue()(),
                        (*parent->cl_buffer.back())(),
                        mapped_ptr,
                        0,
                        nullptr,
                        nullptr);
                }

                parent->cl_subbuffer.push_back(std::vector<cl::Buffer*>());

                parent->cl_image.push_back(nullptr);
            }

            if (parent->buffer_mask & CL_MEM_COPY_HOST_PTR)
                parent->buffer_mask &= ~CL_MEM_COPY_HOST_PTR;
        }
        else
        {
            if(make_image == true) {

                parent->cl_buffer.push_back(nullptr);

                cl::ImageFormat format;

                format.image_channel_order = CL_R;
                format.image_channel_data_type = CL_FLOAT;

                parent->cl_image.push_back( new cl::Image2D( parent->device_context->get_context(),
                                                              parent->buffer_mask,
                                                              format, image_width, image_height, 0 /* pitch auto*/,
                                                              ptr, &err ));

                if( err != 0 ) {
                    THROW_ERROR( err, " Error creating OpenCL fully_connected buffer for inputs , error: " );
                }


            } else {
                parent->cl_buffer.push_back(new cl::Buffer(
                    parent->device_context->get_context(),
                    parent->buffer_mask & ~CL_MEM_COPY_HOST_PTR,
                    parent->buffer_aligned_size,
                    nullptr,
                    &err));

                if (err != CL_SUCCESS)
                    THROW_ERROR(err, "Error creating nn_cl_data buffer.");

                float* mapped_ptr = static_cast<float*>(
                    clEnqueueMapBuffer(
                    parent->device_context->get_command_queue()(),
                    (*parent->cl_buffer[0])(),
                    true,
                    CL_MEM_READ_WRITE,
                    0,
                    parent->buffer_size,
                    0,
                    nullptr,
                    nullptr,
                    &err));

                if (err != CL_SUCCESS)
                    THROW_ERROR(err, "Error in mapping buffer at nn_cl_data buffer creation with HOST COPY.");

                if (parent->buffer_mask & CL_MEM_COPY_HOST_PTR)
                {
                    parent->buffer_mask &= ~CL_MEM_COPY_HOST_PTR;

                    memcpy(mapped_ptr, ptr, parent->buffer_size);
                }
                else
                {
                    // needed for buffers with zero-padding
                    memset(mapped_ptr, 0, parent->buffer_size);
                }

                clEnqueueUnmapMemObject(
                    parent->device_context->get_command_queue()(),
                    (*parent->cl_buffer[0])(),
                    mapped_ptr,
                    0,
                    nullptr,
                    nullptr);

#if 0       // This has to be debugged
                parent->cl_buffer.push_back(new cl::Buffer(
                    parent->device_context->get_context(),
                    parent->buffer_mask, 
                    parent->buffer_aligned_size,
                    ptr,
                    &err));

                if (err != CL_SUCCESS)
                    THROW_ERROR(err, "Error creating nn_cl_data buffer.");

 #endif //0
                parent->cl_subbuffer.push_back(std::vector<cl::Buffer*>());

                parent->cl_image.push_back(nullptr);
            }
        }
    }

    nn_cl_data(
        nn_cl_data& in_data, 
        nn_workload_data_coords_t& coords_begin, 
        nn_workload_data_coords_t& coords_end)
    {
        parent = in_data.parent;

        ++parent->reference_counter;

        assert(coords_begin.dimension == coords_end.dimension);
        const uint32_t dimension_count = 6;

        for (uint32_t i = 0; i < dimension_count; ++i)
        {
            view_begin.t[i] = in_data.view_begin.t[i] + coords_begin.t[i];
            view_end.t[i] = in_data.view_begin.t[i] + coords_end.t[i];
        }
    }

    void add_data_subbuffer(uint64_t index = 0, uint64_t offset = 0, uint64_t data_size = 0)
    {
        cl_buffer_region region = {offset, data_size};
        cl_int err;
        parent->cl_subbuffer[index].push_back(
            new cl::Buffer(parent->cl_buffer[index]->createSubBuffer(
                parent->buffer_mask,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &err)));

        if (err != CL_SUCCESS)
            THROW_ERROR(err, "Error creating nn_cl_data sub-buffer.");
    }

    void convert_buffer_to_image(uint64_t index = 0, uint64_t width = 0, uint64_t height = 0)
    {
        if (width % 64 == 0)
        {
            cl_image_format image_format;
            image_format.image_channel_order = CL_R;
            image_format.image_channel_data_type = CL_FLOAT;

            cl_image_desc image_desc;
            image_desc.num_mip_levels = 0;
            image_desc.num_samples = 0;
            image_desc.image_array_size = 1;
            image_desc.image_width = width;
            image_desc.image_height = height;
            image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            image_desc.image_row_pitch = 0;
            image_desc.image_slice_pitch = 0;
            image_desc.buffer = (*parent->cl_buffer[index])();

            cl_int err;
            parent->cl_image[index] = new cl::Image2D(clCreateImage(
                parent->device_context->get_context()(),
                parent->buffer_mask,
                &image_format,
                &image_desc,
                nullptr, &err));

            if (err != CL_SUCCESS)
                THROW_ERROR(err, "Error converting nn_cl_data buffer to image.");
        }
        else
        {

            cl_int err;
            float* ptr = static_cast<float*>(
                clEnqueueMapBuffer(
                parent->device_context->get_command_queue()(),
                (*parent->cl_buffer[index])(),
                true,
                CL_MEM_READ_ONLY,
                0,
                parent->buffer_aligned_size,
                0,
                nullptr,
                nullptr,
                &err));

            if (err != CL_SUCCESS)
                THROW_ERROR(err, "Error in mapping buffer at INPUT memcpy.");

            cl::ImageFormat image_format;
            image_format.image_channel_order = CL_R;
            image_format.image_channel_data_type = CL_FLOAT;

            parent->cl_image[index] = new cl::Image2D(
                parent->device_context->get_context(),
                parent->buffer_mask | CL_MEM_COPY_HOST_PTR,
                image_format,
                width,
                height,
                0,
                ptr,
                &err);

            if (err != CL_SUCCESS)
                THROW_ERROR(err, "Error converting nn_cl_data buffer to image.");

            clEnqueueUnmapMemObject(
                parent->device_context->get_command_queue()(),
                (*parent->cl_buffer[index])(),
                ptr,
                0,
                nullptr,
                nullptr);

            // Invalidate buffer in this case.
            delete parent->cl_buffer[index];
            parent->cl_buffer[index] = nullptr;
        }

    }

    ~nn_cl_data()
    {
        if (parent)
        {
            --parent->reference_counter;

            if (parent->reference_counter == 0)
            {
                for (auto vector : parent->cl_subbuffer)
                {
                    for (auto ptr : vector)
                    {
                        delete ptr;
                    }
                }

                for (auto ptr : parent->cl_buffer)
                {
                    delete ptr;
                }
                for (auto ptr : parent->cl_image)
                {
                    delete ptr;
                }

                delete parent;
                parent = nullptr;
            }
        }
    }
};


namespace nn {
/* arguments for convolution layers (with shared weights) */
struct arguments_forward_convolution {
    NN_PADDING_MODE             padding;          /* padding mode */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
    uint32_t                    stride[2];        /* stride during filtering operation */
    nn_cl_data                 *biases;
    nn_cl_data                 *weights;
    nn_argument_activation_t    activation;       /* activation data */
};

/* arguments for fully connected layers */
struct arguments_forward_fully_connected {
    nn_cl_data                 *biases;
    nn_cl_data                 *weights;
    nn_argument_activation_t    activation;     /* activation data */
};

/* arguments for merged convolution and pooling layers (with shared weights) */
struct arguments_forward_convolution_pooling_max_2x2_stride_2x2 {
    nn_cl_data                 *biases;
    nn_cl_data                 *weights;
    nn_argument_activation_t    activation;       /* activation data */
    NN_PADDING_MODE             padding;          /* padding mode */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
    uint32_t                    stride[2];        /* stride during filtering operation */
};

/* returns view of an input as an output
   Size of input view is determined from output buffer. Only origin is necessary. */
struct arguments_view {
    uint32_t                    origin[3];      /* left-up-front corner of view */
};

/* arguments for fully connected layers */
struct arguments_forward_arithmetic {
    nn_cl_data                 *factor;
    NN_ARITHMETIC_FUNCTION     arithmetic_func;     /* activation data */
};



} // namespace nn


typedef struct nn_gpu_workload_item {
    /* following fields are copied from work item */
    NN_WORK_ITEM_TYPE         type;               /* work item type */
    nn_cl_data *output;
    std::unique_ptr<nn_cl_data> output_view;

    //std::unique_ptr<nn::nn_data_t< float >>     output;
    std::vector<nn_gpu_workload_item *> input;      /* input array */

    /* union of packaged arguments for each layer type */
    union {
        /* arguments for simple NN layers */
        nn_arguments_input_t                                            input;
        nn_arguments_output_t                                           output;
        nn_arguments_view_t                                             view;
        nn::arguments_forward_arithmetic                                forward_arithmetic;
        nn::arguments_forward_convolution                               forward_convolution;
        nn::arguments_forward_fully_connected                           forward_fully_connected;
        nn_arguments_forward_pooling_t                                  forward_pooling;
        nn_arguments_forward_normalization_t                            forward_normalization;

        /* arguments for for complex NN layers */
        nn::arguments_forward_convolution_pooling_max_2x2_stride_2x2    forward_convolution_pooling_max_2x2_stride_2x2;
    } arguments;

    std::vector<nn_gpu_workload_item *> use;        /* workload items that use result of current one */

    uint32_t output_w_pad_for_next_layer;
    uint32_t output_h_pad_for_next_layer;
    nn_gpu_workload_item( ) : output_w_pad_for_next_layer( 0 ), output_h_pad_for_next_layer( 0 )
    {}

    ~nn_gpu_workload_item()
    {
        // For every workload_item but INPUT and OUTPUT release allocated output
        if(this->type != NN_WORK_ITEM_TYPE_OUTPUT)
        {
            delete output;
        }
        if(this->type == NN_WORK_ITEM_TYPE_ARITHMETIC)
        {
            delete this->arguments.forward_arithmetic.factor; 
        }
        else if(this->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED)
        {
            delete this->arguments.forward_fully_connected.weights; 
            delete this->arguments.forward_fully_connected.biases; 
        } 
        else if (this->type == NN_WORK_ITEM_TYPE_CONVOLUTION)
        {
            delete this->arguments.forward_convolution.weights; 
            delete this->arguments.forward_convolution.biases;
        }
    }
} nn_gpu_workload_item_t;

typedef struct nn_gpu_workload {
    char nn_workload_placeholder[sizeof(struct nn_workload)];
    // Here comes workload_items
    std::vector<nn_gpu_workload_item*> m_workload_items;
} nn_gpu_workload_t;
