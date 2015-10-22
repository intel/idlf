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

//the batch size that is minimal required for usage with jit version
const uint64_t BATCH_ACCEPTED_BLOCK = 24;

//the size of register used for shifting with batch layout / number if pics/floats that are processed at the same time
const uint64_t BATCH_SHIFT = 8;

//number of registers (blocks to process) in the batch format
const uint64_t BATCH_BLOCKS = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT;

//required alignment of all buffers used by jit primitives
const uint64_t BUFFERS_ALIGNMENT = BATCH_SHIFT * sizeof(float);

struct InputHeight {};
struct InputWidth {};
struct InputFeats {};
struct OutputHeight {};
struct OutputWidth {};
struct OutputFeats {};
struct KernelHeight {};
struct KernelWidth {};
struct KernelFeats {};
struct Rows {};
struct Cols {};
struct Batch {};
struct InputFeatsStart {};
struct OutputFeatsStart {};
struct PoolingWidth {};
struct PoolingHeight {};

template <typename T_What, typename T>
struct Value
{
    T value;
    Value(T_What, T value) : value(value) {}

    template <typename T_Other>
    Value(const Value<T_What, T_Other>& other) : value(other.value) {}


    operator T() const { return value; }

    Value<T_What, T> operator+(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value + other.value); }
    Value<T_What, T> operator-(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value - other.value); }
    Value<T_What, T> operator*(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value * other.value); }
    Value<T_What, T> operator/(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value / other.value); }
};
template <typename T_What, typename T> Value<T_What, T> make(T val) { return Value<T_What, T>(T_What(), val); }

struct KernelCenter { Value<Rows, uint64_t> row; Value<Cols, uint64_t> col; };
struct Stride { Value<Rows, uint64_t> rows; Value<Cols, uint64_t> cols; };
struct InputStart { Value<Rows, uint64_t> row; Value<Cols, uint64_t> col; };

template <typename T_Height, typename T_Width>
struct Dimensions2D
{
    Value<T_Height, uint64_t> height;
    Value<T_Width, uint64_t> width;
    Dimensions2D(Value<T_Height, uint64_t> height, Value<T_Width, uint64_t> width)
        : height(height)
        , width(width)
    {}

    uint64_t size() const { return height * width; }
};
template <typename T_Height, typename T_Width, typename T_Feats>
struct Dimensions3D : Dimensions2D<T_Height, T_Width>
{
    Value<T_Feats, uint64_t> feats;
    Dimensions3D(Value<T_Height, uint64_t> height,
                 Value<T_Width, uint64_t> width,
                 Value<T_Feats, uint64_t> feats)
        : Dimensions2D<T_Height, T_Width>(height, width)
        , feats(feats)
    {}
    uint64_t size() const { return Dimensions2D<T_Height, T_Width>::size() * feats; }
};
typedef Dimensions3D<InputHeight, InputWidth, InputFeats> InputDimensions;
typedef Dimensions3D<OutputHeight, OutputWidth, OutputFeats> OutputDimensions;
typedef Dimensions3D<KernelHeight, KernelWidth, KernelFeats> KernelDimensions;
typedef Dimensions2D<PoolingHeight, PoolingWidth> PoolingDimensions;
struct StrideInfo { Stride stride; InputStart start; };
struct KernelInfo {
    KernelDimensions dims;
    Value<OutputFeats, uint64_t> out_feats;
    KernelCenter center;
    Stride stride;
};

struct PoolingInfo {
    PoolingDimensions dims;
    Stride stride;
};

