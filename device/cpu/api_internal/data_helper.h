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

#include "device/cpu/api_internal/cpu_device_internal.h"

namespace nn {

template <nn_workload_data_tag_t T_tag, typename T> struct data_helper;

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, T> {
    static const nn_workload_data_layout_t &layout;

    static nn::workload_data<T> *create(
        nn_device_internal *device,
        const uint32_t x_size,
        const uint32_t y_size,
        const uint32_t z_size,
        const uint32_t n_size,
        const uint32_t block_size,
        const uint32_t padding_left,
        const uint32_t padding_right,
        const uint32_t padding_top,
        const uint32_t padding_bottom) {
        nn_workload_data_coords_t size(
            static_cast<uint32_t>(n_size),
            static_cast<uint32_t>(x_size),
            static_cast<uint32_t>(y_size),
            static_cast<uint32_t>((z_size - 1) / block_size + 1),
            block_size,
            1
        );

        auto buffer = new nn::workload_data<T>(
            NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, size, layout, padding_left, padding_right, padding_top, padding_bottom);

        return buffer;
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::data<T, 4> *destination, const nn::workload_data<T> *source) {
        throw std::logic_error("The method or operation is not implemented.");
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<T, 4> *source) {
        throw std::logic_error("The method or operation is not implemented.");
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_pxyznq();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pxyznq<float>()                     { return nn::layout_t<nn::layout_pxyznq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pxyznq<int16_t>()                   { return nn::layout_t<nn::layout_pxyznq_i16>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pxyznq<nn::layout_zblockxyzn_f32>() { return nn::layout_t<nn::layout_pxyznq_i16>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, T>::layout = data_helper_layout_lookup_pxyznq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_XBLOCKNX, T> {
    static const nn_workload_data_layout_t &layout;

    static nn::workload_data<T> *create(
        nn_device_internal *device,
        const uint32_t z_size,
        const uint32_t n_size,
        const uint32_t block_size) {
        nn_workload_data_coords_t size(
            static_cast<uint32_t>(n_size),
            1,
            1,
            static_cast<uint32_t>((z_size - 1) / block_size + 1),
            block_size,
            1
        );

        auto buffer = new nn::workload_data<T>(
            NN_WORKLOAD_DATA_TAG_XBLOCKNX, size, layout);

        return buffer;
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq<float>()               { return nn::layout_t<nn::layout_pnzxyq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq<int16_t>()             { return nn::layout_t<nn::layout_pnzxyq_i16>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq<int32_t>()             { return nn::layout_t<nn::layout_pnzxyq_i32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq<nn::layout_x2nx_f32>() { return nn::layout_t<nn::layout_pnzxyq_i32>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_XBLOCKNX, T>::layout = data_helper_layout_lookup_pnzxyq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, T> {
    static const nn_workload_data_layout_t &layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta = false>
    static void copy(nn_device_internal *device, nn::data<value_type, 4> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout==layout);

        auto source_buffer = source->parent->data_buffer;
        if(copy_delta){
            source_buffer = source->parent->delta_buffer;
            assert(source_buffer != nullptr);
        }

        const size_t size_z = destination->size[0], size_x = destination->size[1], size_y = destination->size[2],
                     size_n = destination->size[3];

        auto source_length = source->get_length();
        assert(source_length.t[NN_DATA_COORD_z] == size_z);
        assert(source_length.t[NN_DATA_COORD_n] == size_n);

        assert(source_length.t[NN_DATA_COORD_p] == 1);
        assert(source_length.t[NN_DATA_COORD_q] == 1);
        if (source->parent->lengths.t[NN_DATA_COORD_x] == size_x &&
            source->parent->lengths.t[NN_DATA_COORD_y] == size_y) {
            // cover all buffer not only the view

            if (destination->count() == source->parent->buffer_size / sizeof(value_type)) {
                // view matches buffer
                memcpy(destination->buffer, source_buffer, source->parent->buffer_size);
            } else {
                const size_t stride_x = source->parent->lengths.t[NN_DATA_COORD_z],
                             stride_y = stride_x * source->parent->lengths.t[NN_DATA_COORD_x],
                             stride_n = stride_y * source->parent->lengths.t[NN_DATA_COORD_y];

                for (size_t n = 0; n < size_n; ++n)
                    for (size_t y = 0; y < size_y; ++y)
                        for (size_t x = 0; x < size_x; ++x) {
                            void *source_ptr = reinterpret_cast<value_type *>(source_buffer)+
                                               source->view_begin.t[NN_DATA_COORD_z] +
                                               stride_x * x +
                                               stride_y * y +
                                               stride_n * (source->view_begin.t[NN_DATA_COORD_n] + n);
                            memcpy(
                                &destination->at(0, x, y, n), source_ptr, sizeof(value_type) * source_length.t[NN_DATA_COORD_z]);
                        }
            }
        } else {
            // cover only the view]

            assert(source_length.t[NN_DATA_COORD_x] == size_x);
            assert(source_length.t[NN_DATA_COORD_y] == size_y);

            const size_t stride_x = source->parent->lengths.t[NN_DATA_COORD_z],
                         stride_y = stride_x * source->parent->lengths.t[NN_DATA_COORD_x],
                         stride_n = stride_y * source->parent->lengths.t[NN_DATA_COORD_y];

            for (size_t n = 0; n < source_length.t[NN_DATA_COORD_n]; ++n)
                for (size_t y = 0; y < source_length.t[NN_DATA_COORD_y]; ++y)
                    for (size_t x = 0; x < source_length.t[NN_DATA_COORD_x]; ++x) {
                        void *source_ptr = reinterpret_cast<value_type *>(source_buffer)+
                                           source->view_begin.t[NN_DATA_COORD_z] +
                                           stride_x * (source->view_begin.t[NN_DATA_COORD_x] + x) +
                                           stride_y * (source->view_begin.t[NN_DATA_COORD_y] + y) +
                                           stride_n * (source->view_begin.t[NN_DATA_COORD_n] + n);
                        memcpy(&destination->at(0, x, y, n), source_ptr, sizeof(value_type) * source_length.t[NN_DATA_COORD_z]);
                    }
        }
    }

    template <bool copy_delta = false>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 4> *source) {
        assert(destination->parent->layout==layout);

        const size_t size_n = source->size[3], size_x = source->size[1], size_y = source->size[2],
                     size_z = source->size[0];
        auto destination_length = destination->get_length();
        assert(destination_length.t[NN_DATA_COORD_z] == size_z);
        assert(destination_length.t[NN_DATA_COORD_n] == size_n);
        assert(destination_length.t[NN_DATA_COORD_p] == 1);
        assert(destination_length.t[NN_DATA_COORD_q] == 1);

        if (destination->parent->lengths.t[NN_DATA_COORD_x] == size_x &&
            destination->parent->lengths.t[NN_DATA_COORD_y] == size_y) {
            // cover all buffer not only the view

            if (source->count() == destination->parent->buffer_size / sizeof(value_type)) {
                // view matches buffer
                if(copy_delta)
                    memcpy(destination->parent->delta_buffer, source->buffer, destination->parent->buffer_size);
                else
                    memcpy(destination->parent->data_buffer, source->buffer, destination->parent->buffer_size);
            } else {
                const size_t buffer_size_z = destination->parent->lengths.t[NN_DATA_COORD_z],
                             buffer_size_x = destination->parent->lengths.t[NN_DATA_COORD_x],
                             buffer_size_y = destination->parent->lengths.t[NN_DATA_COORD_y];
                auto dest_buffer = copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer;
                for (size_t n = 0u; n < size_n; ++n)
                    for (size_t y = 0u; y < size_y; ++y)
                        for (size_t x = 0u; x < size_x; ++x)
                            for (size_t z = 0u; z < size_z; ++z)
                                ((value_type *)(dest_buffer))
                                    [z +
                                     buffer_size_z *
                                         (x + buffer_size_x * (y + buffer_size_y * n))] =
                                         ((value_type *)(source->buffer))[z + size_z * (x + size_x * (y + size_y * n))];
            }
        }else{
            // cover only the view
            assert(destination_length.t[NN_DATA_COORD_x] == size_x);
            assert(destination_length.t[NN_DATA_COORD_y] == size_y);

            if (source->count() == destination->parent->buffer_size / sizeof(value_type)) {
                // view matches buffer
                memcpy(destination->parent->data_buffer, source->buffer, destination->parent->buffer_size);
            } else {
                const size_t view_begin_x = destination->view_begin.t[NN_DATA_COORD_x],
                             view_begin_y = destination->view_begin.t[NN_DATA_COORD_y],
                             buffer_size_z = destination->parent->lengths.t[NN_DATA_COORD_z],
                             buffer_size_x = destination->parent->lengths.t[NN_DATA_COORD_x],
                             buffer_size_y = destination->parent->lengths.t[NN_DATA_COORD_y];
                for (size_t n = 0u; n < size_n; ++n)
                    for (size_t y = 0u; y < size_y; ++y)
                        for (size_t x = 0u; x < size_x; ++x)
                            for (size_t z = 0u; z < size_z; ++z)
                                ((value_type *)(destination->parent->data_buffer))
                                    [z +
                                     buffer_size_z *
                                         (x + view_begin_x + buffer_size_x * (y + view_begin_y + buffer_size_y * n))] =
                                         ((value_type *)(source->buffer))[z + size_z * (x + size_x * (y + size_y * n))];
            }
        }
    }

    static nn::workload_data<T> *create(nn_device_internal *device,
                                             const uint32_t x_size,
                                             const uint32_t y_size,
                                             const uint32_t i_size,
                                             const uint32_t n_size,
                                             const uint32_t padding_left,
                                             const uint32_t padding_right,
                                             const uint32_t padding_top,
                                             const uint32_t padding_bottom,
                                             const bool allocate_delta) {
        // TODO remove view offset and size params (extra create_view call will be used for those)
        const nn_workload_data_coords_t size(n_size, x_size, y_size, i_size, 1, 1);

        auto buffer = new nn::workload_data<T>(
            NN_WORKLOAD_DATA_TAG_ZXYN, size, layout, padding_left, padding_right, padding_top, padding_bottom, false, allocate_delta);

        return buffer;
    }

    template <bool T_view_supported>
    static bool validate(const ::nn_workload_data_t *data,
                         const uint32_t x_size,
                         const uint32_t y_size,
                         const uint32_t z_size,
                         const uint32_t n_size,
                         const uint32_t min_padding_left,
                         const uint32_t min_padding_right,
                         const uint32_t min_padding_top,
                         const uint32_t min_padding_bottom) {
        if (data->parent->tag != NN_WORKLOAD_DATA_TAG_ZXYN)
            return false;

        if (layout != data->parent->layout)
            return false;

        const auto view_size = static_cast<const nn::workload_data<T> *>(data)->get_length();

        if (view_size.t[NN_DATA_COORD_n] != n_size)
            return false;

        if (view_size.t[NN_DATA_COORD_x] != (x_size))
            return false;
        if (view_size.t[NN_DATA_COORD_y] != y_size)
            return false;
        if (view_size.t[NN_DATA_COORD_z] != z_size)
            return false;

        if (data->view_begin.t[NN_DATA_COORD_x] < min_padding_left)
            return false;
        if (data->view_begin.t[NN_DATA_COORD_y] < min_padding_top)
            return false;

        if (data->parent->lengths.t[NN_DATA_COORD_x] - (data->view_end.t[NN_DATA_COORD_x] + 1) < min_padding_right)
            return false;
        if (data->parent->lengths.t[NN_DATA_COORD_y] - (data->view_end.t[NN_DATA_COORD_y] + 1) < min_padding_bottom)
            return false;

        if (!T_view_supported &&
            (data->parent->buffer_size / data->parent->data_type_size) != x_size * y_size * z_size * n_size)
            return false;

        return true;
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_zxynpq();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_zxynpq<float>()                 { return nn::layout_t<nn::layout_zxynpq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_zxynpq<nn::layout_zxynpq_f32>() { return nn::layout_t<nn::layout_zxynpq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_zxynpq<nn::layout_zxyn_f32>()   { return nn::layout_t<nn::layout_zxynpq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_zxynpq<nn::layout_zxy_f32>()    { return nn::layout_t<nn::layout_zxynpq_f32>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, T>::layout = data_helper_layout_lookup_zxynpq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_ZXY, T> {
    static const nn_workload_data_layout_t &layout;

    using value_type = typename nn::workload_data<T>::item_type;

    // source is XYZ
    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 3> *source) {
        assert(destination->parent->layout == layout);

        const size_t size_x = source->size[0], size_y = source->size[1], size_z = source->size[2];
        auto destination_length = destination->get_length();
        assert(destination_length.t[NN_DATA_COORD_z] == size_z);
        assert(destination_length.t[NN_DATA_COORD_x] == size_x);
        assert(destination_length.t[NN_DATA_COORD_y] == size_y);

        assert(destination_length.t[NN_DATA_COORD_n] == 1);
        assert(destination_length.t[NN_DATA_COORD_p] == 1);
        assert(destination_length.t[NN_DATA_COORD_q] == 1);

        const size_t view_begin_x = destination->view_begin.t[NN_DATA_COORD_x],
                     view_begin_y = destination->view_begin.t[NN_DATA_COORD_y],
                     buffer_size_z = destination->parent->lengths.t[NN_DATA_COORD_z],
                     buffer_size_x = destination->parent->lengths.t[NN_DATA_COORD_x],
                     buffer_size_y = destination->parent->lengths.t[NN_DATA_COORD_y];

        auto dest_buffer = copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer;

        for (size_t y = 0u; y < size_y; ++y)
            for (size_t x = 0u; x < size_x; ++x)
                for (size_t z = 0u; z < size_z; ++z)
                    ((value_type *)(dest_buffer))[z + buffer_size_z * (x + view_begin_x +
                        buffer_size_x * (y + view_begin_y))] = ((value_type *)(source->buffer))[x + size_x * (y + size_y * (z))];
    }

    static nn::workload_data<T> *create(nn_device_internal *device,
                                             const uint32_t x_size,
                                             const uint32_t y_size,
                                             const uint32_t z_size,
                                             const bool allocate_delta) {
        // TODO remove view offset and size params (extra create_view call will be used for those)
        const nn_workload_data_coords_t size(1, x_size, y_size, z_size, 1, 1);

        auto buffer = new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_ZXY, size, layout, false, allocate_delta);

        return buffer;
    }
};
template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_ZXY, T>::layout = data_helper_layout_lookup_zxynpq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_X2NX, T> {
    static const nn_workload_data_layout_t &layout;

    static nn::workload_data<T> *create(nn_device_internal *device, uint32_t z_size, uint32_t n_size, uint32_t block_size) {
        const nn_workload_data_coords_t size(
            n_size,
            1,
            1,
            static_cast<uint32_t>((z_size - 1) / block_size + 1),
            block_size,
            1 );
        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_X2NX, size, layout);
    }
};

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_X2NX, T>::layout = data_helper_layout_lookup_pnzxyq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_NX, T> {
    static const nn_workload_data_layout_t &layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta = false>
    static void copy(nn_device_internal *device, nn::data<value_type, 2> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout==layout);

        const auto x_size = destination->size[0], n_size = destination->size[1];
        auto source_length = source->get_length();
        assert(n_size == source_length.t[NN_DATA_COORD_n]);
        assert(x_size == source_length.t[NN_DATA_COORD_x]);

        assert(source_length.t[NN_DATA_COORD_y] == 1);
        assert(source_length.t[NN_DATA_COORD_z] == 1);
        assert(source_length.t[NN_DATA_COORD_p] == 1);
        assert(source_length.t[NN_DATA_COORD_q] == 1);

        auto dest = static_cast<value_type *>(destination->buffer);
        for (auto n = 0; n < n_size; ++n)
            for (auto x = 0; x < x_size; ++x)
            if(copy_delta)
                *(dest++) = (*const_cast<nn::workload_data<T> *>(source)).delta_at(n, x, 0, 0, 0, 0);
            else
                *(dest++) = (*const_cast<nn::workload_data<T> *>(source))(n, x, 0, 0, 0, 0);
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 2> *source) {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();
        assert(destination_length.t[NN_DATA_COORD_y] == 1);
        assert(destination_length.t[NN_DATA_COORD_z] == 1);
        assert(destination_length.t[NN_DATA_COORD_p] == 1);
        assert(destination_length.t[NN_DATA_COORD_q] == 1);

        const auto n_size = source->size[1], x_size = source->size[0];

        assert(destination_length.t[NN_DATA_COORD_n] == n_size);
        assert(destination_length.t[NN_DATA_COORD_x] == x_size);

        for (auto n = 0; n < n_size; ++n)
            for (auto x = 0; x < x_size; ++x)
                if(copy_delta)
                    destination->delta_at(n, x, 0, 0, 0, 0) = source->at(x, n);
                else
                    (*destination)(n, x, 0, 0, 0, 0) = source->at(x, n);
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 4> *source) {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();
        assert(destination_length.t[NN_DATA_COORD_y] == 1);
        assert(destination_length.t[NN_DATA_COORD_z] == 1);
        assert(destination_length.t[NN_DATA_COORD_p] == 1);
        assert(destination_length.t[NN_DATA_COORD_q] == 1);

        const auto n_size = source->size[3], x_size = source->size[0], y_size = source->size[1],
                   z_size = source->size[2];

        assert(destination_length.t[NN_DATA_COORD_n] == n_size);
        assert(destination_length.t[NN_DATA_COORD_x] == x_size * y_size * z_size);

        for (auto n = 0; n < n_size; ++n)
            for (auto z = 0u; z < z_size; ++z)
                for (auto y = 0u; y < y_size; ++y)
                    for (auto x = 0u; x < x_size; ++x)
                        if(copy_delta)
                            destination->delta_at(
                                static_cast<uint32_t>(n),
                                static_cast<uint32_t>(z + z_size * (x + x_size * y)),
                                0, 0, 0, 0) = source->at(x, y, z, n);
                        else
                            (*destination)(
                                static_cast<uint32_t>(n),
                                static_cast<uint32_t>(z + z_size * (x + x_size * y)),
                                0, 0, 0, 0) = source->at(x, y, z, n);
    }

    static nn::workload_data<T> *create(nn_device_internal *device, uint32_t x_size, uint32_t n_size, bool allocate_delta = false) {
        const nn_workload_data_coords_t size(n_size, x_size, 1, 1, 1, 1);
        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_NX, size, layout, false, allocate_delta);
    }

    static bool validate(const ::nn_workload_data_t *data, const uint32_t x_size, const uint32_t n_size) {
        if (data->parent->tag != NN_WORKLOAD_DATA_TAG_NX)
            return false;

        if (layout != data->parent->layout)
            return false;

        const auto view_size = reinterpret_cast<const workload_data<T> *>(data)->get_length();

        if (view_size.t[NN_DATA_COORD_n] != n_size)
            return false;

        if (view_size.t[NN_DATA_COORD_x] != x_size)
            return false;

        return true;
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq<float>()                 { return nn::layout_t<nn::layout_nxyzpq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq<int16_t>()               { return nn::layout_t<nn::layout_nxyzpq_i16>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq<int32_t>()               { return nn::layout_t<nn::layout_nxyzpq_i32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq<nn::layout_f32>()        { return nn::layout_t<nn::layout_nxyzpq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq<nn::layout_nxyzpq_f32>() { return nn::layout_t<nn::layout_nxyzpq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq<nn::layout_nx_f32>()     { return nn::layout_t<nn::layout_nxyzpq_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_nxyzpq<nn::layout_o_f32>()      { return nn::layout_t<nn::layout_nxyzpq_f32>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_NX, T>::layout = data_helper_layout_lookup_nxyzpq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_OI, T> {
    static const nn_workload_data_layout_t &layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 2> *source) {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();
        assert(destination->parent->buffer_size / sizeof(value_type) == source->count()); // no support for views

        const auto i_size = source->size[0], o_size = source->size[1];

        assert(destination_length.t[NN_DATA_COORD_x] == i_size);
        assert(destination_length.t[NN_DATA_COORD_y] == o_size);

        auto src = static_cast<value_type *>(source->buffer);
        auto dst = static_cast<value_type *>(copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer);

        for (size_t i = 0u; i < i_size; ++i)
            for (size_t o = 0u; o < o_size; ++o)
                *dst++ = src[i + o * i_size];
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::data<value_type, 2> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout == layout);

        auto source_length = source->get_length();
        assert(source->parent->buffer_size / sizeof(value_type) == destination->count()); // no support for views

        const auto i_size = destination->size[0], o_size = destination->size[1];

        assert(source_length.t[NN_DATA_COORD_x] == i_size);
        assert(source_length.t[NN_DATA_COORD_y] == o_size);

        auto src = static_cast<value_type *>(copy_delta ? source->parent->delta_buffer : source->parent->data_buffer);
        auto dst = static_cast<value_type *>(destination->buffer);

        for (size_t i = 0u; i < i_size; ++i)
            for (size_t o = 0u; o < o_size; ++o)
                dst[i + o * i_size] = *src++;
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 4> *source) {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();
        assert(destination->parent->buffer_size / sizeof(value_type) == source->count()); // no support for views

        const auto o_size = source->size[3], x_size = source->size[0], y_size = source->size[1],
                   z_size = source->size[2];

        assert(destination_length.t[NN_DATA_COORD_x] == x_size * y_size * z_size);
        assert(destination_length.t[NN_DATA_COORD_y] == o_size);

        auto dst = static_cast<value_type *>(copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer);
        for (auto y = 0; y < y_size; ++y)
            for (auto x = 0; x < x_size; ++x)
                for (auto z = 0; z < z_size; ++z)
                    for (auto o = 0; o < o_size; ++o)
                        *dst++ = source->at(x, y, z, o);
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::data<value_type, 4> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout == layout);

        auto source_length = source->get_length();
        assert(source->parent->buffer_size / sizeof(value_type) == destination->count()); // no support for views

        const auto o_size = destination->size[3], x_size = destination->size[0], y_size = destination->size[1],
            z_size = destination->size[2];

        assert(source_length.t[NN_DATA_COORD_x] == x_size * y_size * z_size);
        assert(source_length.t[NN_DATA_COORD_y] == o_size);

        auto src = static_cast<value_type *>(copy_delta ? source->parent->delta_buffer : source->parent->data_buffer);
        for(auto y = 0; y < y_size; ++y)
            for(auto x = 0; x < x_size; ++x)
                for(auto z = 0; z < z_size; ++z)
                    for(auto o = 0; o < o_size; ++o)
                        destination->at(x, y, z, o) = *src++;
    }

    static nn::workload_data<T> *create(nn_device_internal *device, uint32_t i_size, uint32_t o_size, bool allocate_delta = false) {
        const nn_workload_data_coords_t size(1, i_size, o_size, 1, 1, 1);
        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_OI, size, layout, false, allocate_delta);
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_yxzpqn();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_yxzpqn<float>()                 { return nn::layout_t<nn::layout_yxzpqn_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_yxzpqn<nn::layout_yxzpqn_f32>() { return nn::layout_t<nn::layout_yxzpqn_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_yxzpqn<nn::layout_oi_f32>()     { return nn::layout_t<nn::layout_yxzpqn_f32>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_OI, T>::layout = data_helper_layout_lookup_yxzpqn<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIO, T> {
    static const nn_workload_data_layout_t &layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 2> *source) {
        assert(destination->parent->layout==layout);

        auto destination_length = destination->get_length();

        // no support for views
        assert(destination_length.t[NN_DATA_COORD_x] == destination->parent->lengths.t[NN_DATA_COORD_x]);
        assert(destination_length.t[NN_DATA_COORD_q] == destination->parent->lengths.t[NN_DATA_COORD_q]);

        const auto i_size = source->size[0], o_size = source->size[1];
        const auto block_size = destination->parent->lengths.t[NN_DATA_COORD_p];

        assert(destination_length.t[NN_DATA_COORD_x] == i_size);
        assert(destination_length.t[NN_DATA_COORD_q] == (o_size - 1) / block_size + 1);

        /*
        Code below this comment is a performance optimized version of:
        auto width = weights.size[0];
        auto height = weights.size[1];
        for(auto x=0u; x<width; ++x)
            for(auto y=0u; y<height; ++y)
                (*result)(0, x, 0, 0, y%C_max_accumulators, y/C_max_accumulators) = weights.at(x,y);
        ...which is left here for readability.
        */
        auto src = static_cast<float *>(source->buffer);
        auto dst = static_cast<float *>(copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer);
        for (auto o = 0u; o < o_size; ++o) {
            auto o_offset = o % block_size + (o / block_size) * i_size * block_size;
            for (auto i = 0u; i < i_size; ++i)
                dst[o_offset + i * block_size] = src[i_size * o + i];
        }
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::data<value_type, 2> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout == layout);

        auto source_length = source->get_length();

        // no support for views
        assert(source_length.t[NN_DATA_COORD_x] == source->parent->lengths.t[NN_DATA_COORD_x]);
        assert(source_length.t[NN_DATA_COORD_q] == source->parent->lengths.t[NN_DATA_COORD_q]);

        const auto i_size = destination->size[0], o_size = destination->size[1];
        const auto block_size = source->parent->lengths.t[NN_DATA_COORD_p];

        assert(source_length.t[NN_DATA_COORD_x] == i_size);
        assert(source_length.t[NN_DATA_COORD_q] == (o_size - 1) / block_size + 1);

        auto src = static_cast<float *>(copy_delta ? source->parent->delta_buffer : source->parent->data_buffer);
        auto dst = static_cast<float *>(destination->buffer);
        for (auto o = 0u; o < o_size; ++o) {
            auto o_offset = o % block_size + (o / block_size) * i_size * block_size;
            for (auto i = 0u; i < i_size; ++i)
                dst[i_size * o + i] = src[o_offset + i * block_size];
        }
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 4> *source) {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();

        // no support for views
        assert(destination_length.t[NN_DATA_COORD_x] == destination->parent->lengths.t[NN_DATA_COORD_x]);
        assert(destination_length.t[NN_DATA_COORD_q] == destination->parent->lengths.t[NN_DATA_COORD_q]);

        const auto o_size = source->size[3], x_size = source->size[0], y_size = source->size[1],
                   z_size = source->size[2];
        const auto block_size = destination->parent->lengths.t[NN_DATA_COORD_p];
        const auto i_size = x_size * y_size * z_size;

        assert(destination_length.t[NN_DATA_COORD_x] == i_size);
        assert(destination_length.t[NN_DATA_COORD_q] == (o_size - 1) / block_size + 1);

        uint32_t last_non_full_slice = o_size % block_size;
        /*
        Code below this comment is a performance optimized version of:
        for (auto z = 0u; z < z_size; ++z)
            for (auto y = 0u; y < y_size; ++y)
                for (auto x = 0u; x < x_size; ++x)
                    for(auto o=0u; o<num_output; ++o)
                        (*result)(0, z + z_size * (x + x_size * y), 0, 0, o%C_max_accumulators, o/C_max_accumulators) =
                            weights.at(x, y, z, o);
        ...which is left here for readability.
        */
        auto src = static_cast<float *>(source->buffer);
        const auto src_y_stride = x_size;
        const auto src_z_stride = y_size * src_y_stride;
        const auto src_o_stride = z_size * src_z_stride;
        auto dst = static_cast<float *>(copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer);
        for (auto z = 0u; z < z_size; ++z)
            for (auto y = 0u; y < y_size; ++y)
                for (auto x = 0u; x < x_size; ++x) {
                    auto dst_xyz_offset = (z + z_size * (x + x_size * y)) * block_size;
                    auto src_xyz_offset = x + y * src_y_stride + z * src_z_stride;
                    for (auto o = 0u; o < o_size; ++o) {
                        auto dp = o % block_size;
                        auto dq = o - dp;
                        dst[dp + dst_xyz_offset + i_size * dq] = src[src_xyz_offset + o * src_o_stride];
                    }
                }
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::data<value_type, 4> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout == layout);

        auto source_length = source->get_length();

        // no support for views
        assert(source_length.t[NN_DATA_COORD_x] == source->parent->lengths.t[NN_DATA_COORD_x]);
        assert(source_length.t[NN_DATA_COORD_q] == source->parent->lengths.t[NN_DATA_COORD_q]);

        const auto o_size = destination->size[3], x_size = destination->size[0], y_size = destination->size[1],
                   z_size = destination->size[2];
        const auto block_size = source->parent->lengths.t[NN_DATA_COORD_p];
        const auto i_size = x_size * y_size * z_size;

        assert(source_length.t[NN_DATA_COORD_x] == i_size);
        assert(source_length.t[NN_DATA_COORD_q] == (o_size - 1) / block_size + 1);

        uint32_t last_non_full_slice = o_size % block_size;

        auto dst = static_cast<float *>(destination->buffer);
        const auto dst_y_stride = x_size;
        const auto dst_z_stride = y_size * dst_y_stride;
        const auto dst_o_stride = z_size * dst_z_stride;
        auto src = static_cast<float *>(copy_delta ? source->parent->delta_buffer : source->parent->data_buffer);
        for (auto z = 0u; z < z_size; ++z)
            for (auto y = 0u; y < y_size; ++y)
                for (auto x = 0u; x < x_size; ++x) {
                    auto src_xyz_offset = (z + z_size * (x + x_size * y)) * block_size;
                    auto dst_xyz_offset = x + y * dst_y_stride + z * dst_z_stride;
                    for (auto o = 0u; o < o_size; ++o) {
                        auto dp = o % block_size;
                        auto dq = o - dp;
                        dst[dst_xyz_offset + o * dst_o_stride] = src[dp + src_xyz_offset + i_size * dq];
                    }
                }
    }

    static nn::workload_data<T> *create(nn_device_internal *device,
                                             uint32_t block_size,
                                             uint32_t i_size,
                                             uint32_t o_size,
                                             bool allocate_delta) {
        const nn_workload_data_coords_t size(1, i_size, 1, 1, block_size, (o_size - 1) / block_size + 1);
        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_OBLOCKIO, size, layout, false, allocate_delta);
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_pzxyqn();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pzxyqn<float>()                     { return nn::layout_t<nn::layout_pzxyqn_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pzxyqn<nn::layout_pzxyqn_f32>()     { return nn::layout_t<nn::layout_pzxyqn_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pzxyqn<nn::layout_oblockixyo_f32>() { return nn::layout_t<nn::layout_pzxyqn_f32>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pzxyqn<nn::layout_oblockio_f32>()   { return nn::layout_t<nn::layout_pzxyqn_f32>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIO, T>::layout = data_helper_layout_lookup_pzxyqn<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIXYO, T> {
    static const nn_workload_data_layout_t &layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 4> *source) {
        assert(destination->parent->layout==layout);

        auto destination_length = destination->get_length();

        // no support for views
        assert(destination_length.t[NN_DATA_COORD_n] == destination->parent->lengths.t[NN_DATA_COORD_n]);
        assert(destination_length.t[NN_DATA_COORD_x] == destination->parent->lengths.t[NN_DATA_COORD_x]);
        assert(destination_length.t[NN_DATA_COORD_y] == destination->parent->lengths.t[NN_DATA_COORD_y]);
        assert(destination_length.t[NN_DATA_COORD_z] == destination->parent->lengths.t[NN_DATA_COORD_z]);
        assert(destination_length.t[NN_DATA_COORD_p] == destination->parent->lengths.t[NN_DATA_COORD_p]);
        assert(destination_length.t[NN_DATA_COORD_q] == destination->parent->lengths.t[NN_DATA_COORD_q]);

        const auto o_size = source->size[3], // number of output feature maps / filters
            x_size = source->size[0],        // kernel width
            y_size = source->size[1],        // kernel height
            i_size = source->size[2];        // number of input feature maps
        const auto block_size = destination->parent->lengths.t[NN_DATA_COORD_p];
        const auto block_count = (o_size - 1) / block_size + 1;

        assert(destination_length.t[NN_DATA_COORD_x] == x_size);
        assert(destination_length.t[NN_DATA_COORD_y] == y_size);
        assert(destination_length.t[NN_DATA_COORD_z] == i_size);
        assert(destination_length.t[NN_DATA_COORD_q] == block_count);

        /*
        Code below this comment is a performance optimized version of:
        for (size_t q = 0u; q < size.t[5]; ++q)
            for (size_t p = 0u; p < size.t[4]; ++p)
                for (size_t z = 0u; z < size.t[3]; ++z)
                    for (size_t y = 0u; y < size.t[2]; ++y)
                        for (size_t x = 0u; x < size.t[1]; ++x)
                            //              n, x, y, z, p, q  =            x, y, i, o
                            (*load_weights)(0, x, y, z, p, q) = weights.at(x, y, z, q * C_slice_size + p);
        ...which is left here for readability.
        */

        auto src = static_cast<float *>(source->buffer);
        auto dst = static_cast<float *>(copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer);
        auto src_stride_y = x_size;
        auto src_stride_i = src_stride_y * y_size;
        auto src_stride_o = src_stride_i * i_size;
        for (size_t q = 0u; q < block_count; ++q)
            for (size_t y = 0u; y < y_size; ++y)
                for (size_t x = 0u; x < x_size; ++x)
                    for (size_t i = 0u; i < i_size; ++i)
                        for (size_t p = 0u; p < block_size; ++p) {
                            if(q * block_size + p < o_size) {
                                *(dst++) = src[x + src_stride_y * y + src_stride_i * i + src_stride_o * (q * block_size + p)];
                            } else {
                                *(dst++) = 0.0f;
                            }
                        }
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::data<value_type, 4> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout == layout);

        auto source_length = source->get_length();

        // no support for views
        assert(source_length.t[NN_DATA_COORD_n] == source->parent->lengths.t[NN_DATA_COORD_n]);
        assert(source_length.t[NN_DATA_COORD_x] == source->parent->lengths.t[NN_DATA_COORD_x]);
        assert(source_length.t[NN_DATA_COORD_y] == source->parent->lengths.t[NN_DATA_COORD_y]);
        assert(source_length.t[NN_DATA_COORD_z] == source->parent->lengths.t[NN_DATA_COORD_z]);
        assert(source_length.t[NN_DATA_COORD_p] == source->parent->lengths.t[NN_DATA_COORD_p]);
        assert(source_length.t[NN_DATA_COORD_q] == source->parent->lengths.t[NN_DATA_COORD_q]);

        const auto o_size = destination->size[3], // number of output feature maps / filters
            x_size = destination->size[0],        // kernel width
            y_size = destination->size[1],        // kernel height
            i_size = destination->size[2];        // number of input feature maps
        const auto block_size = source->parent->lengths.t[NN_DATA_COORD_p];
        const auto block_count = (o_size - 1) / block_size + 1;

        assert(source_length.t[NN_DATA_COORD_x] == x_size);
        assert(source_length.t[NN_DATA_COORD_y] == y_size);
        assert(source_length.t[NN_DATA_COORD_z] == i_size);
        assert(source_length.t[NN_DATA_COORD_q] == block_count);

        auto src = static_cast<float *>(copy_delta ? source->parent->delta_buffer : source->parent->data_buffer);
        auto dst = static_cast<float *>(destination->buffer);
        auto dst_stride_y = x_size;
        auto dst_stride_i = dst_stride_y * y_size;
        auto dst_stride_o = dst_stride_i * i_size;
        for (size_t q = 0u; q < block_count; ++q)
            for (size_t y = 0u; y < y_size; ++y)
                for (size_t x = 0u; x < x_size; ++x)
                    for (size_t i = 0u; i < i_size; ++i)
                        for(size_t p = 0u; p < block_size; ++p)
                            dst[x + dst_stride_y * y + dst_stride_i * i + dst_stride_o * (q * block_size + p)] = *(src++);
    }

    static nn::workload_data<T> *create(nn_device_internal *device,
                                             uint32_t block_size,
                                             uint32_t x_size,
                                             uint32_t y_size,
                                             uint32_t i_size,
                                             uint32_t o_size,
                                             bool allocate_delta) {
        const nn_workload_data_coords_t size(1, x_size, y_size, i_size, block_size, (o_size - 1) / block_size + 1);
        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_OBLOCKIXYO, size, layout, false, allocate_delta);
    }
};

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIXYO, T>::layout = data_helper_layout_lookup_pzxyqn<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_O, T> {
    static const nn_workload_data_layout_t &layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<value_type, 1> *source) {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();
        assert(destination->parent->buffer_size / sizeof(value_type) == source->count()); // no support for views

        const auto o_size = source->size[0]; // number of output feature maps / filters

        assert(destination_length.t[NN_DATA_COORD_x] == o_size);

        for (uint32_t o = 0u; o < o_size; ++o) {
            if (copy_delta)
                destination->delta_at(0, o, 0, 0, 0, 0) = source->at(o);
            else
                (*destination)(0, o, 0, 0, 0, 0) = source->at(o);
        }
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::data<value_type, 1> *destination, const nn::workload_data<T> *source) {
        assert(source->parent->layout == layout);

        auto source_length = source->get_length();
        assert(source->parent->buffer_size / sizeof(value_type) == destination->count()); // no support for views

        const auto o_size = destination->size[0]; // number of output feature maps / filters

        assert(source_length.t[NN_DATA_COORD_x] == o_size);

        for(uint32_t o = 0u; o < o_size; ++o) {
            if(copy_delta)
                destination->at(o) = source->delta_at(0, o, 0, 0, 0, 0);
            else
                destination->at(o) = source->at(0, o, 0, 0, 0, 0);
        }
    }
    static nn::workload_data<T> *create(nn_device_internal *device, uint32_t o_size, bool allocate_delta) {
        const nn_workload_data_coords_t size( 1, o_size, 1, 1, 1, 1 );
        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_O, size, layout, false, allocate_delta);
    }
};
template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_O, T>::layout = data_helper_layout_lookup_nxyzpq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_I2O32IXYO, T> {
    static const nn_workload_data_layout_t &layout;

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<T, 4> *source) {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();

        const unsigned int OFMpBlock = 2;
        const unsigned int o_block_size = 32;

        auto dst = static_cast<int16_t *>(destination->parent->data_buffer);
        auto src = static_cast<int16_t *>(source->buffer);

        for (auto q = 0u; q < destination_length.t[5]; ++q)
            for (auto x = 0u; x < destination_length.t[1]; ++x)
                for (auto n = 0u; n < destination_length.t[0]; ++n)
                    for (auto z = 0u; z < destination_length.t[3]; ++z)
                        for (auto p = 0u; p < destination_length.t[4]; ++p)
                            for (auto y = 0u; y < destination_length.t[2]; ++y)
                                *(dst++) = (z * OFMpBlock + y < source->size[2])
                                    ? src[n + source->size[0] * (x + source->size[1] * ((z * OFMpBlock + y) + source->size[2] * (q * o_block_size + p)))]
                                    : 0;
    }

    static nn::workload_data<T> *create(nn_device_internal *device,
        uint32_t block_size,
        uint32_t x_size,
        uint32_t y_size,
        uint32_t i_size,
        uint32_t o_size) {

        const unsigned int i_block_size = 2;
        const unsigned int o_block_size = 32;

        const nn_workload_data_coords_t size(
            x_size,
            y_size,
            i_block_size,
            (i_size - 1) / i_block_size + 1,
            o_block_size,
            (o_size - 1) / o_block_size + 1
        );

        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_I2O32IXYO, size, layout);
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_ypznxq();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_ypznxq<int16_t>() { return nn::layout_t<nn::layout_ypznxq_i16>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_I2O32IXYO, T>::layout = data_helper_layout_lookup_ypznxq<T>();

template <typename T> struct data_helper<NN_WORKLOAD_DATA_TAG_I2O8IO, T> {
    static const nn_workload_data_layout_t &layout;

    static const uint32_t i_block_size = 2;
    static const uint32_t o_block_size = 8;

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<T, 4> *source)
    {
        assert(destination->parent->layout==layout);

        auto destination_length = destination->get_length();

        const auto num_inputs = source->size[0] * source->size[1] * source->size[2];

        auto dst = static_cast<int16_t *>(destination->parent->data_buffer);
        auto src = static_cast<int16_t *>(source->buffer);

        for (std::uint32_t p = 0; p < destination_length.t[4]; ++p)
            for (std::uint32_t y = 0; y < destination_length.t[2]; ++y)
                for (std::uint32_t z = 0; z < destination_length.t[3]; ++z)
                    for (std::uint32_t x = 0; x < destination_length.t[1]; ++x)
                        *(dst++) = src[(y * i_block_size + x) + (p * o_block_size + z) * num_inputs];
    }

    template <bool copy_delta>
    static void copy(nn_device_internal *device, nn::workload_data<T> *destination, const nn::data<T, 2> *source)
    {
        assert(destination->parent->layout == layout);

        auto destination_length = destination->get_length();

        const auto num_inputs = source->size[0];

        auto dst = static_cast<int16_t *>(destination->parent->data_buffer);
        auto src = static_cast<int16_t *>(source->buffer);

        for (std::uint32_t p = 0; p < destination_length.t[4]; ++p)
            for (std::uint32_t y = 0; y < destination_length.t[2]; ++y)
                for (std::uint32_t z = 0; z < destination_length.t[3]; ++z)
                    for (std::uint32_t x = 0; x < destination_length.t[1]; ++x)
                        *(dst++) = src[(y * i_block_size + x) + (p * o_block_size + z) * num_inputs];
    }

    static nn::workload_data<T> *create(
        nn_device_internal *device,
        uint32_t block_size,
        uint32_t i_size,
        uint32_t o_size) {

        const nn_workload_data_coords_t size(
            1,
            i_block_size,
            (i_size - 1) / i_block_size + 1,
            o_block_size,
            (o_size - 1) / o_block_size + 1,
            1
        );

        return new nn::workload_data<T>(NN_WORKLOAD_DATA_TAG_I2O8IO, size, layout);
    }
};

template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_xzynpq();
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_xzynpq<int16_t>() { return nn::layout_t<nn::layout_xzynpq_i16>::layout; }
template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_xzynpq<int32_t>() { return nn::layout_t<nn::layout_xzynpq_i32>::layout; }

template <typename T>
const nn_workload_data_layout_t &data_helper<NN_WORKLOAD_DATA_TAG_I2O8IO, T>::layout = data_helper_layout_lookup_xzynpq<T>();

template <typename T>
struct data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIOXY, T> {
    static const nn_workload_data_layout_t& layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta>
    static void copy(nn_device_internal *device,
                     nn::workload_data<T> *destination,
                     const nn::data<value_type, 4> *source)
    {

        assert(destination->parent->layout==layout);

        auto destination_length = destination->get_length();

        // no support for views
        assert(destination_length.t[NN_DATA_COORD_n] == destination->parent->lengths.t[NN_DATA_COORD_n]);
        assert(destination_length.t[NN_DATA_COORD_x] == destination->parent->lengths.t[NN_DATA_COORD_x]);
        assert(destination_length.t[NN_DATA_COORD_y] == destination->parent->lengths.t[NN_DATA_COORD_y]);
        assert(destination_length.t[NN_DATA_COORD_z] == destination->parent->lengths.t[NN_DATA_COORD_z]);
        assert(destination_length.t[NN_DATA_COORD_p] == destination->parent->lengths.t[NN_DATA_COORD_p]);
        assert(destination_length.t[NN_DATA_COORD_q] == destination->parent->lengths.t[NN_DATA_COORD_q]);

        const auto o_size = source->size[3], // number of output feature maps / filters
            x_size = source->size[0],        // kernel width
            y_size = source->size[1],        // kernel height
            i_size = source->size[2];        // number of input feature maps
        const auto block_size = destination->parent->lengths.t[NN_DATA_COORD_p];
        const auto block_count = (o_size - 1) / block_size + 1;

        assert(destination_length.t[NN_DATA_COORD_x] == x_size);
        assert(destination_length.t[NN_DATA_COORD_y] == y_size);
        assert(destination_length.t[NN_DATA_COORD_z] == i_size);
        assert(destination_length.t[NN_DATA_COORD_q] == block_count);

        auto src = static_cast<float *>(source->buffer);
        auto dst = static_cast<float *>(copy_delta ? destination->parent->delta_buffer : destination->parent->data_buffer);
        for (size_t y = 0u; y < y_size; ++y)
            for (size_t x = 0u; x < x_size; ++x)
                for (size_t q = 0u; q < block_count; ++q)
                    for (size_t i = 0u; i < i_size; ++i)
                        for (size_t p = 0u; p < block_size; ++p)
                            *(dst++) = src[x + x_size * (y + y_size * (i + i_size * (q * block_size + p)))];
    }

    static nn::workload_data<T> *create(uint32_t width,
                                        uint32_t height,
                                        uint32_t in_feats,
                                        uint32_t out_feats,
                                        uint32_t out_feats_block_size,
                                        bool = false) {
        uint32_t out_feats_blocks = (out_feats + out_feats_block_size - 1) / out_feats_block_size;
        return new nn::workload_data<T>(
            NN_WORKLOAD_DATA_TAG_OBLOCKIOXY,
            nn_workload_data_coords_t(1, width, height, in_feats, out_feats_block_size, out_feats_blocks),
            layout);
    }
};

template <typename T>
const nn_workload_data_layout_t& data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIOXY, T>::layout = nn::layout_t<nn::layout_oblockioxy_f32>::layout;

template <typename T>
struct data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, T> {
    static const nn_workload_data_layout_t& layout;

    using value_type = typename nn::workload_data<T>::item_type;

    template <bool copy_delta>
    static void copy(nn_device_internal *device,
                     nn::workload_data<T> *destination,
                     const nn::data<value_type, 4> *source)
    {
        throw std::runtime_error("unimplemented");
    }

    static nn::workload_data<T> *create(uint32_t batch,
                                        uint32_t width,
                                        uint32_t height,
                                        uint32_t in_feats,
                                        uint32_t batch_block_size,
                                        bool = false) {
        uint32_t blocks = (batch + batch_block_size - 1) / batch_block_size;
        return new nn::workload_data<T>(
            NN_WORKLOAD_DATA_TAG_NBLOCKZXYN,
            nn_workload_data_coords_t(blocks, width, height, in_feats, batch_block_size, 1),
            layout);

    }
};

template <typename T>
const nn_workload_data_layout_t& data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, T>::layout = nn::layout_t<nn::layout_nblockzxyn_f32> ::layout;

} //namespace nn

