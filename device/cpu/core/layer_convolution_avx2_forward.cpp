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
#include "layer_convolution_avx2_forward.h"
#include "device/common/nn_asmjit_compilation.h"
#include <cassert>
#include <cstdlib>
#include <string>
#include <tuple>
#include <cstdlib>
#include <functional>
#include <vector>
#include <thread>
#include <iostream>
#include <immintrin.h>
#include "jit_conv_generic.h"
#include <map>
#include <set>

namespace convolution
{
namespace forward
{

struct StackWithData
{
//args
    float*      input;
    float*      output;
    float*      filter;
    float*      bias;

    uint64_t    output_col;
    uint64_t    output_row;

    uint64_t    kernel_row_begin;
    uint64_t    kernel_row_end;
    uint64_t    kernel_col_begin;
    uint64_t    kernel_col_end;

    uint64_t    in_row_shift;
    uint64_t    in_col_shift;
};
typedef void (*CompiledKernelConvolutionFunc)(StackWithData*);
struct CompiledKernelConvolution
{
    CompiledKernelConvolutionFunc asmjit_conv;
    jit_convolution* xbyak_conv;
};

using namespace nn::asm_compilation;

struct Mod { ValueU64<Rows> row; ValueU64<Cols> col; };
struct BeginAdd { ValueU64<Rows> rows; ValueU64<Cols> cols; };


typedef std::function<void(asmjit::X86Assembler& a)> CalcFunc;
CalcFunc operator and(CalcFunc left, CalcFunc right)
{
    return [=](asmjit::X86Assembler& a){ left(a); right(a); };
}

namespace
{

float ZERO = 0;

uint64_t ceil_div(uint64_t arg, uint64_t div)
{
    return (arg + div - 1) / div;
}

const uint64_t BUFFER_ALIGNMENT = 4096u;
const uint64_t ACC_REGISTERS = 12u;
const uint64_t BATCH_SHIFT = BUFFERS_ALIGNMENT;
const uint64_t BATCH_BLOCKS = ACC_REGISTERS / OUTPUT_ACCEPTED_MOD;
const uint64_t BATCH = BATCH_SHIFT / sizeof(float) *  BATCH_BLOCKS;
const uint64_t BATCH_SIZE = BATCH * sizeof(float);
static_assert(BATCH == BATCH_ACCEPTED_BLOCK, "invalid batch sizes");

typedef void (*CompiledConvolve)(StackWithData*);

void createInternalForInOutBlock(asmjit::X86Assembler& a,
                                 asmjit::X86GpReg filterReg,
                                 asmjit::X86GpReg inputReg,
                                 uint64_t numOfIfs,
                                 uint64_t numOfOfs)
{
    using namespace asmjit;
    using namespace asmjit::x86;
    for (uint64_t i = 0u; i < numOfIfs; ++i)
    {
        for (uint64_t j = 0; j < BATCH_BLOCKS; ++j)
            a.vmovaps(ymm(ACC_REGISTERS + j), ptr(inputReg, (BATCH_BLOCKS * i + j) * BATCH_SHIFT));
        for (uint64_t j = 0u; j < numOfOfs; ++j)
        {
            X86Mem filterPtr = ptr(filterReg, (j + i * OUTPUT_ACCEPTED_MOD) * sizeof(float));
            a.vbroadcastss(ymm15, filterPtr);
            for (uint64_t k = 0; k < BATCH_BLOCKS; ++k)
                a.vfmadd231ps(ymm(BATCH_BLOCKS * j + k), ymm(ACC_REGISTERS + k), ymm15);
        }
    }
}

void createConvolve_batch24_singleMult(
    asmjit::X86Assembler& a,
    asmjit::X86GpReg inputReg,
    asmjit::X86GpReg filterReg,
    asmjit::X86GpReg infeatBlockReg,
    uint64_t infeat_block_size,
    uint64_t infeats,
    uint64_t outfeats)
{
    using namespace asmjit;
    using namespace asmjit::x86;
    using namespace std::placeholders;

    loop(infeatBlockReg, 0, infeats / infeat_block_size,
        std::bind(createInternalForInOutBlock, _1, filterReg, inputReg, infeat_block_size, outfeats)
        and
        [&](asmjit::X86Assembler& a){
            a.add(inputReg, infeat_block_size * BATCH_SIZE);
            a.add(filterReg, sizeof(float) * OUTPUT_ACCEPTED_MOD * infeat_block_size);
        }
    )(a);
    if (infeats % infeat_block_size != 0)
        createInternalForInOutBlock(a, filterReg, inputReg, infeats % infeat_block_size, outfeats);
}

template <typename T_Dest, typename T_Base, typename T_Row, typename T_Col>
CalcFunc calcPtr(T_Dest dest, T_Base base, T_Row row, uint64_t width, T_Col col, uint64_t elemSize)
{
    /*
        dest = base + (row * width + col) * elemSize;
    */
    return [=](asmjit::X86Assembler& a){
            a.mov(dest, row);
            a.imul(dest, width);
            a.add(dest, col);
            a.imul(dest, elemSize);
            a.add(dest, base);
        };
}

typedef std::function<void(asmjit::X86Assembler&,
                           asmjit::X86GpReg,
                           asmjit::X86GpReg,
                           asmjit::X86GpReg,
                           uint64_t)> InternalFunc;

void createConvolve_input_batch16_if16n(
    asmjit::X86Assembler& a,
    KernelInfo kernel_info,
    bool apply_relu)
{
    using namespace asmjit;
    using namespace asmjit::x86;
    using namespace std::placeholders;

    uint64_t outfeat_blocks = kernel_info.out_feats / OUTPUT_ACCEPTED_MOD;
    uint64_t total_outfeat_blocks = ceil_div(kernel_info.out_feats, OUTPUT_ACCEPTED_MOD);

    auto accBlockPtr = nn_asmjit_param_ptr(StackWithData, output);
    auto inputBlockPtr = nn_asmjit_param_ptr(StackWithData, input);
    auto filterBlockPtr = nn_asmjit_param_ptr(StackWithData, filter);
    auto biasBlockPtr = nn_asmjit_param_ptr(StackWithData, bias);

    auto kernel_row_begin = nn_asmjit_param_ptr(StackWithData, kernel_row_begin);
    auto kernel_row_end = nn_asmjit_param_ptr(StackWithData, kernel_row_end);
    auto kernel_col_begin = nn_asmjit_param_ptr(StackWithData, kernel_col_begin);
    auto kernel_col_end = nn_asmjit_param_ptr(StackWithData, kernel_col_end);

    auto in_row_shift = nn_asmjit_param_ptr(StackWithData, in_row_shift);
    auto in_col_shift = nn_asmjit_param_ptr(StackWithData, in_col_shift);

    auto out_row = nn_asmjit_param_ptr(StackWithData, output_row);
    auto out_col = nn_asmjit_param_ptr(StackWithData, output_col);

    auto out_block = r11;
    auto in_block = r12;

    auto accPtr = r13;
    auto inputPtr = r14;
    auto kernelPtr = r15;
    auto biasPtr = r10;

    auto kern_row = r8;
    auto kern_col = r9;

    auto calc_input_ptr = [&](X86Assembler& a) {
            a.mov(inputPtr, inputBlockPtr);
            a.mov(rax, kern_row);
            a.mov(rdx, in_row_shift);
            a.mul(rdx);
            a.add(inputPtr, rax);
            a.mov(rax, kernel_info.center.row);
            a.mov(rdx, in_row_shift);
            a.mul(rdx);
            a.sub(inputPtr, rax);

            a.mov(rax, kern_col);
            a.mov(rdx, in_col_shift);
            a.mul(rdx);
            a.add(inputPtr, rax);
            a.mov(rax, kernel_info.center.col);
            a.mov(rdx, in_col_shift);
            a.mul(rdx);
            a.sub(inputPtr, rax);
        };
    auto calc_acc_ptr = [&](X86Assembler& a) {
            a.mov(accPtr, accBlockPtr);

            a.mov(rax, out_block);
            a.mov(rdx, OUTPUT_ACCEPTED_MOD * BATCH_SIZE);
            a.mul(rdx);
            a.add(accPtr, rax);
        };
    auto calc_kernel_ptr = calcPtr(kernelPtr, filterBlockPtr, kern_row, kernel_info.dims.width, kern_col,
            total_outfeat_blocks * OUTPUT_ACCEPTED_MOD *  kernel_info.dims.feats * sizeof(float))
        and [&](X86Assembler& a) {
            a.mov(rax, out_block);
            a.mov(rdx, sizeof(float) * OUTPUT_ACCEPTED_MOD * kernel_info.dims.feats);
            a.mul(rdx);
            a.add(kernelPtr, rax);
        };
    auto calc_bias_ptr = [&](X86Assembler& a) {
        a.mov(biasPtr, biasBlockPtr);
        a.mov(rax, out_block);
        a.mov(rdx, OUTPUT_ACCEPTED_MOD * sizeof(float));
        a.mul(rdx);
        a.add(biasPtr, rax);
    };

    auto prepare_ptrs_and_accs = [&](uint64_t output_feats_to_calculate) {
            assert(output_feats_to_calculate > 0);
            assert(output_feats_to_calculate <= OUTPUT_ACCEPTED_MOD);
            return calc_input_ptr
            and calc_acc_ptr
            and calc_kernel_ptr
            and CalcFunc([=](X86Assembler& a) {
                Label prepared(a);
                Label not_begin(a);
                a.cmp(kern_row, kernel_row_begin);
                a.ja(not_begin);

                a.cmp(kern_col, kernel_col_begin);
                a.ja(not_begin);

                calc_bias_ptr(a);

                for (uint64_t i = 0u; i < output_feats_to_calculate; ++i)
                    for (uint64_t k = 0; k < BATCH_BLOCKS; ++k)
                        a.vbroadcastss(ymm(BATCH_BLOCKS * i + k), ptr(biasPtr, i * sizeof(float)));
                a.jmp(prepared);

                a.bind(not_begin);
                for (uint64_t i = 0u; i < output_feats_to_calculate * BATCH_BLOCKS; ++i)
                    a.vmovaps(ymm(i), ptr(accPtr, i * BATCH_SHIFT));
                a.bind(prepared);
           });
        };
    auto storeAccs = [&](uint64_t output_feats_to_calculate) {
            assert(output_feats_to_calculate > 0);
            assert(output_feats_to_calculate <= OUTPUT_ACCEPTED_MOD);
            return CalcFunc([=](X86Assembler& a){
                    Label not_end(a);

                    if (apply_relu)
                    {
                        a.mov(rax, kern_row);
                        a.inc(rax);
                        a.cmp(rax, kernel_row_end);
                        a.jb(not_end);

                        a.mov(rax, kern_col);
                        a.inc(rax);
                        a.cmp(rax, kernel_col_end);
                        a.jb(not_end);

                        a.mov(rax, (size_t)&ZERO);
                        a.vbroadcastss(ymm15, ptr(rax));
                        for (uint64_t i = 0u; i < output_feats_to_calculate * BATCH_BLOCKS; ++i)
                            a.vmaxps(ymm(i), ymm(i), ymm15);
                    }

                    a.bind(not_end);
                    for (uint64_t i = 0u; i < output_feats_to_calculate * BATCH_BLOCKS; ++i)
                        a.vmovaps(ptr(accPtr, i * BATCH_SHIFT), ymm(i));
                });
        };
    auto out_block_calc = [&](uint64_t output_feats_to_calculate) {
            if (output_feats_to_calculate == 0) return CalcFunc([&](X86Assembler&){});
            return prepare_ptrs_and_accs(output_feats_to_calculate)
            and CalcFunc(std::bind(createConvolve_batch24_singleMult, _1,
                inputPtr, kernelPtr, in_block, INPUT_ACCEPTED_MOD, kernel_info.dims.feats, output_feats_to_calculate))
            and CalcFunc(storeAccs(output_feats_to_calculate));
        };

    loop(kern_row, kernel_row_begin, kernel_row_end,
        loop(kern_col, kernel_col_begin, kernel_col_end,
            loop(out_block, 0, outfeat_blocks, out_block_calc(OUTPUT_ACCEPTED_MOD))
            and
            out_block_calc(kernel_info.out_feats % OUTPUT_ACCEPTED_MOD)
        )
    )(a);
}

void convolve_fst_impl(
    CompiledKernelConvolution* func,
    float* input,
    float* output,
    float* kernel,
    float* bias,
    ValueU64<Rows> out_row,
    ValueU64<Cols> out_col,
    KernelInfo kernel_info,
    StrideInfo stride,
    InputDimensions in_dims,
    OutputDimensions out_dims,
    ValueU64<InputFeatsStart> infeats_window_start,
    ValueU64<OutputFeatsStart> outfeats_window_start)
{
    assert(((uint64_t)input) % BATCH_SHIFT == 0);
    assert(((uint64_t)output) % BATCH_SHIFT == 0);
    assert(((uint64_t)kernel) % BATCH_SHIFT == 0);

    uint64_t in_row_base = out_row * stride.stride.rows + stride.start.row;
    uint64_t in_col_base = out_col * stride.stride.cols + stride.start.col;
    if (in_row_base >= in_dims.height) return;
    if (in_col_base >= in_dims.width) return;

    auto calc_end = [](uint64_t in_size, uint64_t in_base, uint64_t kern_size, uint64_t kern_start) {
            return std::min(in_size - in_base + kern_start, kern_size);
        };
    uint64_t kern_row_end = calc_end(in_dims.height, in_row_base, kernel_info.dims.height, kernel_info.center.row);
    uint64_t kern_col_end = calc_end(in_dims.width, in_col_base, kernel_info.dims.width, kernel_info.center.col);

    auto calc_begin = [](uint64_t in_base, uint64_t kern_size, uint64_t kern_start) {
            return std::max(kern_start, in_base) - in_base;
        };
    uint64_t kern_row_begin = calc_begin(in_row_base, kernel_info.dims.height, kernel_info.center.row);
    uint64_t kern_col_begin = calc_begin(in_col_base, kernel_info.dims.width, kernel_info.center.col);

    uint64_t in_col_shift = in_dims.feats * BATCH_SIZE;
    uint64_t in_row_shift = in_col_shift * in_dims.width;

    auto base_input =
        input + ((in_row_base * in_dims.width + in_col_base) * in_dims.feats + infeats_window_start) * BATCH;
    auto acc =
        output + ((out_row * out_dims.width + out_col) * out_dims.feats + outfeats_window_start) * BATCH;
    StackWithData data = {
        base_input,
        acc,
        kernel,
        bias,
        out_col, out_row,
        kern_row_begin, kern_row_end, kern_col_begin, kern_col_end,
        in_row_shift, in_col_shift};
    func->asmjit_conv(&data);
}

struct SingleInputSize {};
struct SingleOutputSize {};
struct ThreadArgs
{
    CompiledKernelConvolution* func;
    float* input;
    float* output;
    float* kernel;
    float* bias;
    uint64_t size_of_single_input;
    uint64_t size_of_single_output;
    ValueU64<Batch> batch;
    ValueU64<Rows> out_row;
    ValueU64<Cols> out_col;
    KernelInfo kernel_info;
    InputDimensions in_dims;
    OutputDimensions out_dims;
    StrideInfo stride;
    ValueU64<InputFeatsStart> infeats_window_start;
    ValueU64<OutputFeatsStart> outfeats_window_start;

    ThreadArgs(CompiledKernelConvolution* func = nullptr,
               float* input = nullptr,
               float* output = nullptr,
               float* kernel = nullptr,
               float* bias = nullptr,
               uint64_t size_of_single_input = 0,
               uint64_t size_of_single_output = 0,
               ValueU64<Batch> batch = make<Batch>(0),
               ValueU64<Rows> out_row = make<Rows>(0),
               ValueU64<Cols> out_col = make<Cols>(0),
               KernelInfo kernel_info = {
                    {make<KernelHeight>(0), make<KernelWidth>(0), make<KernelFeats>(0)},
                    make<OutputFeats>(0),
                    {make<Rows>(0), make<Cols>(0)},
                    {make<Rows>(0), make<Cols>(0)}},
               InputDimensions in_dims = {make<InputHeight>(0), make<InputWidth>(0), make<InputFeats>(0)},
               OutputDimensions out_dims = {make<OutputHeight>(0), make<OutputWidth>(0), make<OutputFeats>(0)},
               StrideInfo stride = {{make<Rows>(0), make<Cols>(0)}, {make<Rows>(0), make<Cols>(0)}},
               ValueU64<InputFeatsStart> infeats_window_start = make<InputFeatsStart>(0),
               ValueU64<OutputFeatsStart> outfeats_window_start = make<OutputFeatsStart>(0))
        : func(func)
        , input(input)
        , output(output)
        , bias(bias)
        , kernel(kernel)
        , size_of_single_input(size_of_single_input)
        , size_of_single_output(size_of_single_output)
        , batch(batch)
        , out_row(out_row)
        , out_col(out_col)
        , kernel_info(kernel_info)
        , in_dims(in_dims)
        , out_dims(out_dims)
        , stride(stride)
        , infeats_window_start(infeats_window_start)
        , outfeats_window_start(outfeats_window_start)
    {}
};


void convolve_fst_multiple_single_thread(void* arg)
{
    auto args = *static_cast<ThreadArgs*>(arg);
    for (uint64_t batch = 0; batch < args.batch; ++batch)
    {
        convolve_fst_impl(args.func, args.input, args.output, args.kernel, args.bias,
            args.out_row, args.out_col,
            args.kernel_info, args.stride, args.in_dims, args.out_dims,
            args.infeats_window_start,
            args.outfeats_window_start);
        args.input += args.size_of_single_input;
        args.output += args.size_of_single_output;
    }
}

} //namespace

CompiledKernelConvolution* compileConvolution(
    uint64_t num_of_threads,
    KernelInfo kernel_info,
    InputDimensions input_dimensions,
    InputDimensions full_input_dimensions,
    OutputDimensions output_dimensions,
    OutputDimensions full_output_dimensions,
    StrideInfo stride,
    ValueU64<InputFeatsStart> infeats_window_start,
    ValueU64<OutputFeatsStart> outfeats_window_start,
    uint64_t batch_iterations,
    float* input,
    float* output,
    float* weights,
    float* bias,
    bool apply_relu)
{

    if ((kernel_info.dims.width == kernel_info.dims.height)
        and (kernel_info.dims.feats % jit_convolution_generic:: input_features_per_iteration == 0)
        and (kernel_info.out_feats % jit_convolution_generic::output_features_per_iteration == 0))
    {
        return new CompiledKernelConvolution{
            nullptr,
            new jit_convolution_generic(
                batch_iterations,
                apply_relu,
                num_of_threads,

                output,
                full_output_dimensions.width,
                full_output_dimensions.height,
                full_output_dimensions.feats,
                output_dimensions.width,
                output_dimensions.height,
                output_dimensions.feats,
                outfeats_window_start,
                
                input,
                full_input_dimensions.width,
                full_input_dimensions.height,
                full_input_dimensions.feats,
                kernel_info.dims.feats,
                stride.start.col,
                stride.start.row,
                infeats_window_start,

                stride.stride.cols,
                stride.stride.rows,
                
                weights,
                kernel_info.dims.width,
                kernel_info.dims.height,
                kernel_info.center.col,
                kernel_info.center.row,

                bias,
                1,
                1)};
    }
    else
    {
        auto compile = [&](asmjit::X86Assembler& a) { return createConvolve_input_batch16_if16n(a, kernel_info, apply_relu); };
        auto func = (CompiledKernelConvolutionFunc)nn::asm_compilation::asmjit_compile(compile);
        return new CompiledKernelConvolution{func, nullptr};
    }
}

void convolve_fst_threaded_batch(
    nn_thread_worker_pool& thread_pool,
    CompiledKernelConvolution* func,
    float* input,
    float* output,
    float* kernel,
    float* bias,
    ValueU64<Batch> batch,
    KernelInfo kernel_info,
    InputDimensions in_dims,
    OutputDimensions out_dims,
    StrideInfo stride,
    ValueU64<InputFeatsStart> infeats_window_start,
    ValueU64<OutputFeatsStart> outfeats_window_start)
{
    assert(batch % BATCH == 0);
    if (thread_pool.get_num_threads() <= 1)
        return convolve_fst_single_threaded(
            func, input, output, kernel, bias,
            batch, kernel_info, in_dims, out_dims, stride,
            infeats_window_start, outfeats_window_start);

    if (func->xbyak_conv)
    {
        for (auto job : func->xbyak_conv->jobs)
            thread_pool.push_job(job);
    }
    else
    {
        auto size_of_single_input = make<SingleInputSize>(in_dims.size() * BATCH);
        auto size_of_single_output = make<SingleOutputSize>(out_dims.size() * BATCH);

        std::vector<ThreadArgs> jobs_args(out_dims.width * out_dims.height);
        std::vector<nn_multithreaded_request> jobs(jobs_args.size());
        for (uint64_t j = 0u; j < out_dims.width; ++j)
        {
            for (uint64_t i = 0u; i < out_dims.height; ++i)
            {
                uint64_t index = i * out_dims.width + j;
                jobs_args[index] = ThreadArgs(
                        func, input, output, kernel, bias,
                        size_of_single_input, size_of_single_output,
                        make<Batch>(batch / BATCH),
                        make<Rows>(i), make<Cols>(j),
                        kernel_info, in_dims, out_dims, stride,
                        infeats_window_start, outfeats_window_start);

                jobs[index] = nn_multithreaded_request{
                    &convolve_fst_multiple_single_thread, (void*)&jobs_args[index]};
            }
        }
        thread_pool.push_job(jobs);
    }
}

void convolve_fst_single_threaded(
    CompiledKernelConvolution* func,
    float* input,
    float* output,
    float* kernel,
    float* bias,
    ValueU64<Batch> batch,
    KernelInfo kernel_info,
    InputDimensions in_dims,
    OutputDimensions out_dims,
    StrideInfo stride,
    ValueU64<InputFeatsStart> infeats_window_start,
    ValueU64<OutputFeatsStart> outfeats_window_start)
{
    assert(batch % BATCH == 0);
    if (func->xbyak_conv)
    {
        for (auto job : func->xbyak_conv->jobs)
            for (auto task : job)
                task.callback(task.request_handle);
    }
    else
    {
        uint64_t size_of_single_input = in_dims.size() * BATCH;
        uint64_t size_of_single_output = out_dims.size() * BATCH;

        for (uint64_t i = 0u; i < batch / BATCH; ++i)
        {
            for (uint64_t j = 0u; j < out_dims.width; ++j)
                for (uint64_t i = 0u; i < out_dims.height; ++i)
                    convolve_fst_impl(
                        func, input, output, kernel, bias,
                        make<Rows>(i), make<Cols>(j),
                        kernel_info, stride, in_dims, out_dims,
                        infeats_window_start, outfeats_window_start);
            input += size_of_single_input;
            output += size_of_single_output;
        }
    }
}

void release(CompiledKernelConvolution* func)
{
    if (func->asmjit_conv)
        nn::asm_compilation::release(func->asmjit_conv);
    if (func->xbyak_conv)
        delete(func->xbyak_conv);
    delete(func);
}

} //namespace forward
} //namespace convolution

