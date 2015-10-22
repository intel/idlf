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

#include "gtest/gtest.h"

#include "device/api/nn_device_api.h"
#include "device/common/nn_workload_data.h"
#include "device/api/nn_device_interface_0.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "device/cpu/core/layer_arithmetic_operation.h"
#include "device/cpu/core/helper_zxyn_f32.h"

#include <random>
#include <vector>

namespace {
    // ------------------------------------------------------------------------------------------------------
    // Helper classess and functions
    bool compare_work_items(
        nn_workload_item* &work_item,
        nn_workload_item* &work_item_ref ) {

        for(uint32_t batch = 0; batch < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_n]; ++batch)
            for(uint32_t z = 0; z < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_z]; ++z) {
                for(uint32_t x = 0; x < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_x]; ++x) {
                    for(uint32_t y = 0; y < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_y]; ++y)
                    {
                        float value = nn_workload_data_get<float>( work_item->output[0], batch, x, y, z, 0, 0 );
                        float value_ref = nn_workload_data_get<float>( work_item_ref->output[0], batch, x, y, z, 0, 0 );

                        float diff = fabs( value_ref - value );

                        if( value_ref == 0.0f || value == 0.0f || diff < FLT_MIN)
                            if((diff > FLT_MIN) || (diff / value_ref > 5.2e-05F)) {
                                std::cerr << "Position nxyz: " << batch << "\t" << x << "\t" << y << "\t" << z << std::endl
                                        << "Ref val: " << value_ref << std::endl << "Normal val: " << value << std::endl;
                            return false;
                            }
                    }
                }
            }
        return true;
    }
    bool run_work_item(
        nn_workload_item* &work_item,
        bool is_ref,
        NN_ARITHMETIC_FUNCTION function,
        nn_device_t *device ) {

        if(is_ref) {
            // Naive implementation.
            auto *input = &(work_item->input[0]);
            auto output = static_cast<nn::workload_data<nn::layout_f32>*>(work_item->output[0]);

            auto batchsize = output->parent->lengths.t[NN_DATA_COORD_n];
            auto size_x = output->parent->lengths.t[NN_DATA_COORD_x];
            auto size_y = output->parent->lengths.t[NN_DATA_COORD_y];
            auto size_z = output->parent->lengths.t[NN_DATA_COORD_z];

            auto factor = nn::workload_data_cast<nn::layout_f32>(work_item->parameters[0]);
            output = static_cast<nn::workload_data<nn::layout_f32>*>(work_item->output[0]);

         // check sizes  (x,y,z of factor must be equal to x,y,z of input)
           if(size_x != factor->get_length( 1 ) || size_y != factor->get_length( 2 ) || size_z != factor->get_length( 3 ))  
                throw NN_DATA_STATUS_ERROR_INVALID_PARAMETERS; 
            
            switch(function) {
            case NN_ARITHMETIC_FUNCTION_ADDITION: {
                for(unsigned n = 0; n<batchsize; ++n)
                    for(unsigned y = 0; y<size_y; ++y)
                        for(unsigned x = 0; x<size_x; ++x)
                            for(unsigned z = 0; z<size_z; ++z)
                                (*output)(n, x, y, z, 0, 0) = nn_workload_data_get<float>( input->item->output[0], n, x, y, z, 0, 0 ) + (*factor)(0, x, y, z, 0, 0);
            } break;
            case NN_ARITHMETIC_FUNCTION_SUBTRACTION: {
                for(unsigned n = 0; n<batchsize; ++n)
                    for(unsigned y = 0; y<size_y; ++y)
                        for(unsigned x = 0; x<size_x; ++x)
                            for(unsigned z = 0; z<size_z; ++z)
                                (*output)(n, x, y, z, 0, 0) = nn_workload_data_get<float>( input->item->output[0], n, x, y, z, 0, 0 ) - (*factor)(0, x, y, z, 0, 0);
            } break;
            case NN_ARITHMETIC_FUNCTION_MULTIPLICATION: {
                for(unsigned n = 0; n<batchsize; ++n)
                    for(unsigned y = 0; y<size_y; ++y)
                        for(unsigned x = 0; x<size_x; ++x)
                            for(unsigned z = 0; z<size_z; ++z)
                                (*output)(n, x, y, z, 0, 0) = nn_workload_data_get<float>( input->item->output[0], n, x, y, z, 0, 0 ) * (*factor)(0, x, y, z, 0, 0);
            } break;
            case NN_ARITHMETIC_FUNCTION_DIVISION: {
                for(unsigned n = 0; n<batchsize; ++n)
                    for(unsigned y = 0; y<size_y; ++y)
                        for(unsigned x = 0; x<size_x; ++x)
                            for(unsigned z = 0; z<size_z; ++z)
                                if((*factor)(0, x, y, z, 0, 0) != 0.0)
                                    (*output)(n, x, y, z, 0, 0) = nn_workload_data_get<float>( input->item->output[0], n, x, y, z, 0, 0 ) / (*factor)(0, x, y, z, 0, 0);
                                else
                                    (*output)(n, x, y, z, 0, 0) = nn_workload_data_get<float>( input->item->output[0], n, x, y, z, 0, 0 ) > 0 ? std::numeric_limits<float>::infinity()  : -std::numeric_limits<float>::infinity();

            } break;
            default: std::cerr << "Unsupported function, skipping";
            }
            // end of naive implementation
        } else {
            // Use optimized routine.
            work_item->primitive->forward(
                {work_item->input[0].get_data_view()}, {work_item->parameters[0]}, {work_item->output[0]});
        }
        return true;
    }

    void destroy_work_item(
        nn_workload_item* &work_item ) 
    {
        for(auto& parameter : work_item->parameters)
        {
            delete parameter;
            parameter = nullptr;
        }

        for(auto& output : work_item->output)
        {
            delete output;
            output = nullptr;
        }

        delete work_item;
        work_item = nullptr;
    }

    void create_and_initialize_input_item(
        nn_workload_item* &work_item,
        uint32_t input_width,
        uint32_t batch_size,
        uint32_t z ) {

        nn_workload_data_coords_t in_out_coords = {
            batch_size,
            input_width,
            input_width,
            z,
            1,
            1
        };

        work_item = new nn_workload_item();
        work_item->type = NN_WORK_ITEM_TYPE_INPUT;
        work_item->primitive = nullptr;
        work_item->arguments.input.index = 0;

        nn_workload_data_layout_t inp_out_layout = nn::layout_t<nn::layout_zxynpq_f32>::layout;

        work_item->output.push_back(new nn::workload_data<>( in_out_coords, inp_out_layout ));

        for(uint32_t batch = 0; batch < batch_size; ++batch) {
            for(uint32_t input_element = 0; input_element < input_width; ++input_element) {
                float value = 0.03125f;
                value *= pow( 1.01f, input_element );
                value *= pow( 1.01f, batch );
                if(input_element % 2) value *= -1.0f;
                nn_workload_data_get<float>( work_item->output[0], batch, input_element, 0, 0, 0, 0 ) = value;
            }
        }
    }

    void create_and_initialize_work_item(
        nn_workload_item* &work_item,
        nn_workload_item* input_item,
        uint32_t input_width,
        uint32_t batch_size,
        uint32_t z,
        NN_ARITHMETIC_FUNCTION function,
        nn_device_t *device ) {

        nn_workload_data_coords_t in_out_coords = {
            batch_size,
            input_width,
            input_width,
            z,
            1,
            1
        };

        work_item = new nn_workload_item();

        work_item->type = NN_WORK_ITEM_TYPE_ARITHMETIC;
        work_item->primitive = new layer::arithmetic_f32(
            input_width, input_width, z, function, batch_size, reinterpret_cast<nn_device_internal *>(device));

        nn_workload_data_layout_t inp_out_layout = nn::layout_t<nn::layout_zxynpq_f32>::layout;
      
        work_item->input.push_back( {input_item, 0} );
  
        work_item->output.emplace_back(new nn::workload_data<>( in_out_coords, inp_out_layout ));

        for(uint32_t batch = 0; batch < batch_size; ++batch)
            for(uint32_t size_x = 0; size_x < input_width; ++size_x) 
                for(uint32_t size_y = 0; size_y < input_width; ++size_y)
                    for(uint32_t size_z = 0; size_z < z; ++size_z)
                        nn_workload_data_get<float>( work_item->output[0], batch, size_x, size_y, size_z, 0, 0 ) = 0.0f;

        auto factor = new nn::workload_data<nn::layout_f32>(in_out_coords, inp_out_layout);
        work_item->parameters.push_back(factor);

        for(uint32_t batch = 0; batch < batch_size; ++batch)
            for(uint32_t size_x = 0; size_x < input_width; ++size_x)
                for(uint32_t size_y = 0; size_y < input_width; ++size_y)
                    for(uint32_t size_z = 0; size_z < z; ++size_z)
                        (*factor)(batch, size_x, size_y, size_z, 0, 0) = size_x * 0.5;
    }

    bool ult_perform_test(
        uint32_t input_width,
        uint32_t batch_size,
        uint32_t z_size,
        NN_ARITHMETIC_FUNCTION function) 
    {
        bool return_value = true;

        nn_device_description_t device_description;
        nn_device_interface_0_t device_interface_0;

        nn_device_load( &device_description );
        nn_device_interface_open( 0, &device_interface_0 );
        // Input item.
        nn_workload_item* input_item = nullptr;
        create_and_initialize_input_item( input_item, input_width, batch_size, z_size );

        // Work item.
        nn_workload_item* work_item = nullptr;
        create_and_initialize_work_item( work_item, input_item, input_width, batch_size, z_size, function, device_interface_0.device );

        // Reference workload item.
        nn_workload_item* work_item_ref = nullptr;
        create_and_initialize_work_item( work_item_ref, input_item, input_width, batch_size, z_size, function, nullptr );

        // Run items.
        return_value &= run_work_item( work_item, false, function, device_interface_0.device );
        return_value &= run_work_item( work_item_ref, true, function, nullptr );
    
        // Compare results.
        return_value &= compare_work_items( work_item, work_item_ref );
    
        // Cleanup.
        destroy_work_item( work_item_ref );
        destroy_work_item( work_item );
        destroy_work_item( input_item );

        nn_device_interface_close( &device_interface_0 );
        nn_device_unload();

        return return_value;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST( cpu_arithmetic_float, cpu_arithmetic_avx2_vs_naive_normals ) {
    NN_ARITHMETIC_FUNCTION function_tab[] = {
       //           NN_ARITHMETIC_FUNCTION::NN_ARITHMETIC_FUNCTION_NONE,
                  NN_ARITHMETIC_FUNCTION::NN_ARITHMETIC_FUNCTION_ADDITION,
                  NN_ARITHMETIC_FUNCTION::NN_ARITHMETIC_FUNCTION_SUBTRACTION,
                  NN_ARITHMETIC_FUNCTION::NN_ARITHMETIC_FUNCTION_MULTIPLICATION,
                  NN_ARITHMETIC_FUNCTION::NN_ARITHMETIC_FUNCTION_DIVISION,
       //           NN_ARITHMETIC_FUNCTION::NN_ARITHMETIC_FUNCTION_LAST,
    };

     for(auto function : function_tab)
        for(auto batch : { 1, 8, 48 })
            for(auto z : { 1, 3, 7 }) 
                for(auto width : { 5, 32 })  
                    EXPECT_EQ( true, ult_perform_test(
                    width,         // input/output width
                    batch,         // batch size
                    z,
                    function
                    ) ) << "\tfuntion: " << function << "\tbatch: " << batch << "\tz_size: " << z << "\twidth: " << width << std::endl;
}