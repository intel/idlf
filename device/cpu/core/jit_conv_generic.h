#pragma once

#include <functional>
#include <set>
#include <map>
#include <random>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "convolution_jit.h"
#include "device/common/nn_allocate.h"
#include "xbyak/xbyak.h"

// job set of jit convolutions ////////////////////////////////////////////////

namespace detail
{

struct FinishJob
{
    float* output_dest;
    std::vector<float*> outputs;
    uint64_t ofeats;
};

inline void finish_with_relu(void* ptr)
{
    const uint64_t batch_size                    = 24;
    const uint64_t register_width_in_float       = 8;
    const uint64_t registers_in_batch            = batch_size / register_width_in_float;

    auto job = static_cast<FinishJob*>(ptr);
    auto dest = job->output_dest;
    for (auto j = 0u; j < job->outputs.size() - 1; ++j)
    {
        auto curr_src = job->outputs[j];
        auto curr_dest = dest;
        for (auto i = 0u; i < job->ofeats; ++i)
        {
            for (auto j = 0u; j < registers_in_batch; ++j)
            {
                auto acc = _mm256_load_ps(curr_dest);
                acc = _mm256_add_ps(acc, _mm256_load_ps(curr_src));
                _mm256_store_ps(curr_dest, acc);
                curr_dest += register_width_in_float;
                curr_src += register_width_in_float;
            }
        }
    }
    auto curr_src = job->outputs[job->outputs.size() - 1];
    auto curr_dest = dest;
    auto zero = _mm256_setzero_ps();
    for (auto i = 0u; i < job->ofeats; ++i)
    {
        for (auto j = 0u; j < registers_in_batch; ++j)
        {
            auto acc = _mm256_load_ps(curr_dest);
            acc = _mm256_add_ps(acc, _mm256_load_ps(curr_src));
            acc = _mm256_max_ps(acc, zero);
            _mm256_store_ps(curr_dest, acc);
            curr_dest += register_width_in_float;
            curr_src += register_width_in_float;
        }
    }
}

inline void finish_without_relu(void* ptr)
{
    const uint64_t batch_size                    = 24;
    const uint64_t register_width_in_float       = 8;
    const uint64_t registers_in_batch            = batch_size / register_width_in_float;

    auto job = static_cast<FinishJob*>(ptr);
    auto dest = job->output_dest;
    for (auto src : job->outputs)
    {
        auto curr_src = src;
        auto curr_dest = dest;
        for (auto i = 0u; i < job->ofeats; ++i)
        {
            for (auto j = 0u; j < registers_in_batch; ++j)
            {
                auto acc = _mm256_load_ps(curr_dest);
                acc = _mm256_add_ps(acc, _mm256_load_ps(curr_src));
                _mm256_store_ps(curr_dest, acc);
                curr_dest += register_width_in_float;
                curr_src += register_width_in_float;
            }
        }
    }
}

} //namespace detail

struct jit_convolution_generic : public jit_convolution
{
#pragma pack(push, 1)
    struct op_data_t
    {
        float *output;
        float *input;
        float *filter;
        float *bias;
        int8_t type; // 0:init, 1:normal, 2:finalize
    };

    struct op_array_t
    {
        uint64_t count; 
        op_data_t *array;
    };
#pragma pack(pop)

    static const uint64_t input_features_per_iteration  = 8;
    static const uint64_t output_features_per_iteration = 4;
    static const uint64_t batch_size                    = 24;
    static const uint64_t register_width_in_float       = 8;
    static const uint64_t registers_in_batch            = batch_size / register_width_in_float;

    class jit_code : public Xbyak::CodeGenerator
    {
        jit_code &operator=(const jit_code&) = delete;

        void preamble()
        {
            for(auto &reg : scalar_registers_to_preserve)
                push(reg);
            for(auto &reg : vector_registers_to_preserve) {
                auto offset = (&reg - &vector_registers_to_preserve[0]+1)*16;
                vmovdqu(ptr [rsp-offset], reg);
            }
        }

        void postamble()
        {
            for(auto &reg : reverse(vector_registers_to_preserve)) {
                auto offset = (&reg - &vector_registers_to_preserve[0]+1)*16;
                vmovdqu(reg, ptr [rsp-offset]);
            }
            for(auto &reg : reverse(scalar_registers_to_preserve))
                pop(reg);
            ret();
        }

        std::vector<Xbyak::Reg64> scalar_registers_to_preserve;
        std::vector<Xbyak::Xmm> vector_registers_to_preserve;
    public:

        jit_code(uint64_t input_features,
                 uint64_t output_features,
                 bool generate_bias_and_store = true,
                 bool generate_load_and_store = true,
                 bool generate_load_and_relu_store = true,
                 bool generate_bias_and_relu_store = false,
                 bool generate_zero_and_store = false,
                 void* code_ptr = nullptr,
                 size_t code_size = 4 * Xbyak::DEFAULT_MAX_CODE_SIZE)
            : Xbyak::CodeGenerator(code_size, code_ptr)
            , scalar_registers_to_preserve({nn_jit_dangerous_reg1, nn_jit_dangerous_reg2, r11, r12, r13, r14, r15})
            , vector_registers_to_preserve({xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15})
        {
            using Xbyak::Ymm;

            preamble();

            auto output                   = r10;
            auto input                    = r11;
            auto filter                   = r12;
            auto bias                     = r13;
            auto left_jobs                = r14;
            auto job                      = r15;
            auto left_output_feats_blocks = r8;
            auto left_input_feats_blocks  = r9;

            auto aux_input   = nn_jit_dangerous_reg1;
            auto aux_filter  = nn_jit_dangerous_reg2;

            // lambda generating code for one operation:
            // for each block of 4 output feature maps
            //     execute 'prologue'
            //     for each block of input 
            //         process one internal block
            auto code_op_type = [&](std::string tag, std::function<void()> prologue, std::function<void()> epilogue)
            {
                auto code_op_type_block = [&](int n)
                {
                    vmovaps(ymm12, ptr [aux_input + (n * registers_in_batch + 0) * register_width_in_float * sizeof(float)]);
                    vmovaps(ymm13, ptr [aux_input + (n * registers_in_batch + 1) * register_width_in_float * sizeof(float)]);
                    vmovaps(ymm14, ptr [aux_input + (n * registers_in_batch + 2) * register_width_in_float * sizeof(float)]);
                    vbroadcastss(ymm15, ptr [aux_filter + (n * output_features_per_iteration + 0) * sizeof(float)]);
                        vfmadd231ps(ymm0, ymm12, ymm15);
                        vfmadd231ps(ymm1, ymm13, ymm15);
                        vfmadd231ps(ymm2, ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [aux_filter + (n * output_features_per_iteration + 1) * sizeof(float)]);
                        vfmadd231ps(ymm3, ymm12, ymm15);
                        vfmadd231ps(ymm4, ymm13, ymm15);
                        vfmadd231ps(ymm5, ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [aux_filter + (n * output_features_per_iteration + 2) * sizeof(float)]);
                        vfmadd231ps(ymm6, ymm12, ymm15);
                        vfmadd231ps(ymm7, ymm13, ymm15);
                        vfmadd231ps(ymm8, ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [aux_filter + (n * output_features_per_iteration + 3) * sizeof(float)]);
                        vfmadd231ps(ymm9, ymm12, ymm15);
                        vfmadd231ps(ymm10, ymm13, ymm15);
                        vfmadd231ps(ymm11, ymm14, ymm15);
                };
                auto output_feature_loop_tag = "out_feats" + tag;
                auto input_feature_loop_tag  = "in_feats_" + tag;

                align(4);
                mov(left_output_feats_blocks, output_features / output_features_per_iteration);
                L(output_feature_loop_tag);
                    mov(aux_filter, filter);
                    mov(aux_input, input);
                    mov(left_input_feats_blocks, input_features / input_features_per_iteration);

                    prologue();

                    align(4);
                    L(input_feature_loop_tag);
                        for (int n = 0; n < input_features_per_iteration; ++n)
                            code_op_type_block(n);
                        add(aux_input, sizeof(float) * batch_size * input_features_per_iteration);
                        add(aux_filter, sizeof(float) * output_features_per_iteration * input_features_per_iteration);
                        dec(left_input_feats_blocks);
                    jnz(input_feature_loop_tag);

                    epilogue();

                    add(output, output_features_per_iteration * batch_size * sizeof(float));
                    add(filter, output_features_per_iteration * input_features * sizeof(float));
                    add(bias, output_features_per_iteration * sizeof(float));
                    dec(left_output_feats_blocks);
                jnz(output_feature_loop_tag);
                add(job, sizeof(op_data_t));
                dec(left_jobs);
                jnz("op_loop");
            };

            const auto accumulators_count   = output_features_per_iteration * batch_size / register_width_in_float;
            auto code_prologue_load         = [&]{
                for (int n = 0; n < accumulators_count; ++n)
                    vmovaps(Ymm(n),  ptr [output + n * register_width_in_float * sizeof(float)]);
            };
            auto code_prologue_bias         = [&]{
                for(int n = 0; n < accumulators_count; ++n)
                    vbroadcastss(Ymm(n), ptr [bias + (n / registers_in_batch) * sizeof(float)]);
            };
            auto code_prologue_zero = [&]{
                for (int n = 0; n < accumulators_count; ++n)
                    vxorps(Ymm(n), Ymm(n), Ymm(n));
            };
            auto code_epilogue_store        = [&]{
                for (int n = 0; n < accumulators_count; ++n)
                    vmovaps(ptr [output + n * register_width_in_float * sizeof(float)], Ymm(n));
            };
            auto code_epilogue_relu_store   = [&]{
                vxorps(ymm15, ymm15, ymm15);
                for (int n = 0; n < accumulators_count; ++n)
                    vmaxps(Ymm(n), Ymm(n), ymm15);
                for (int n = 0; n < accumulators_count; ++n)
                    vmovaps(ptr[output + n * register_width_in_float * sizeof(float)], Ymm(n));
            };

            // <- code starts here
            // "init" ops: initial zeroing of accumulators (instead of loading them)
            mov(job, ptr [nn_jit_param_reg + offsetof(op_array_t, array)]);
            mov(left_jobs, ptr [nn_jit_param_reg + offsetof(op_array_t, count)]);

            // loop over all operations
            align(4);
            L("op_loop");
                mov(output, ptr [job + offsetof(op_data_t, output)]);
                mov(input,  ptr [job + offsetof(op_data_t, input)]);
                mov(filter, ptr [job + offsetof(op_data_t, filter)]);
                mov(bias,   ptr [job + offsetof(op_data_t, bias)]);

                if (generate_load_and_store
                    and generate_bias_and_store
                    and generate_load_and_relu_store)
                {
                    cmp(byte [job + offsetof(op_data_t, type)], 1);
                    jne("op_type_bias_or_relu", T_NEAR);

                    code_op_type("1", code_prologue_load, code_epilogue_store);
                    jmp("end", T_NEAR);

                    L("op_type_bias_or_relu");
                    cmp(byte [job + offsetof(op_data_t, type)], 2);
                    je("op_type_relu", T_NEAR);

                    code_op_type("2", code_prologue_bias, code_epilogue_store);
                    jmp("end", T_NEAR);

                    L("op_type_relu");
                    code_op_type("0", code_prologue_load, code_epilogue_relu_store);
                }
                else if (generate_zero_and_store and generate_load_and_store)
                {
                    cmp(byte[job + offsetof(op_data_t, type)], 1);
                    jne("op_type_zero", T_NEAR);

                    code_op_type("1", code_prologue_load, code_epilogue_store);
                    jmp("end", T_NEAR);

                    L("op_type_zero");
                    code_op_type("0", code_prologue_zero, code_epilogue_store);
                }
                else if (generate_bias_and_store and generate_load_and_store)
                {
                    cmp(byte[job + offsetof(op_data_t, type)], 1);
                    jne("op_type_bias", T_NEAR);

                    code_op_type("1", code_prologue_load, code_epilogue_store);
                    jmp("end", T_NEAR);

                    L("op_type_bias");
                    code_op_type("0", code_prologue_bias, code_epilogue_store);
                }

            L("end");

            postamble();
        };
    };

    std::vector<std::vector<op_data_t>> op_data;
    std::vector<op_array_t> op_array;
    jit_code code;

    jit_convolution_generic(
        uint64_t batch_iterations,
        bool apply_relu,
        uint64_t num_of_threads,
        
        float*   output,
        uint64_t full_output_width,
        uint64_t full_output_height,
        uint64_t full_output_feature_maps,
        uint64_t output_width,
        uint64_t output_height,
        uint64_t output_feature_maps,
        uint64_t output_begin_z,

        float*   input,
        uint64_t full_input_width,
        uint64_t full_input_height,
        uint64_t full_input_feature_maps,
        uint64_t input_feature_maps,
        uint64_t input_begin_x,
        uint64_t input_begin_y,
        uint64_t input_begin_z,

        uint64_t stride_width,
        uint64_t stride_height,

        float *  filter,
        uint64_t filter_width,
        uint64_t filter_height,
        uint64_t filter_offset_x,
        uint64_t filter_offset_y,
        
        float*   bias,

        uint64_t block_width,
        uint64_t block_height)
            : code(input_feature_maps, output_feature_maps)
            , code_bias(input_feature_maps, output_feature_maps)
    {
        assert(input_feature_maps%input_features_per_iteration==0 && "input feature map count is not a multiple of features-per-iteration");
        assert(output_feature_maps%output_features_per_iteration==0 && "output feature map count is not a multiple of features-per-iteration");

        auto get_output_block = [&](uint64_t b, uint64_t x, uint64_t y) {
                return output
                    + (((b * full_output_height + y) * full_output_width + x)
                        * full_output_feature_maps + output_begin_z) * BATCH_ACCEPTED_BLOCK;
            };
        auto get_input_block = [&](uint64_t b, uint64_t x, uint64_t y) {
                return input
                    + (((b * full_input_height + y) * full_input_width + x)
                        * full_input_feature_maps + input_begin_z) * BATCH_ACCEPTED_BLOCK;
            };
        auto get_filter_block = [&](uint64_t x, uint64_t y) {
                auto output_feature_maps_in_filter =
                    ((output_feature_maps + output_features_per_iteration - 1)
                        / output_features_per_iteration) * output_features_per_iteration;
                return filter
                    + (y * filter_width + x) * output_feature_maps_in_filter * input_feature_maps; 
            };

        typedef std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> JobDescription;
        auto get_batch = [](JobDescription descr) { return std::get<4>(descr); };
        auto get_output_y = [](JobDescription descr) { return std::get<0>(descr); };
        auto get_output_x = [](JobDescription descr) { return std::get<1>(descr); };
        auto get_input_y = [](JobDescription descr) { return std::get<2>(descr); };
        auto get_input_x = [](JobDescription descr) { return std::get<3>(descr); };
        auto make_job_descr = [](uint64_t b, uint64_t oy, uint64_t ox, uint64_t iy, uint64_t ix){
            return std::make_tuple(oy, ox, iy, ix, b); };

        auto ceil_div = [](uint64_t arg, uint64_t div) { return (arg + div - 1) / div; };
        auto get_smallest_thread_sqrt = [](uint64_t arg) {
                auto i = 1u;
                for (; i * i < arg; ++i);
                return i;
            };
        auto ys_per_job_row = 1;
        auto xs_per_job_col = 1;
        auto thread_rows = ceil_div(output_height, ys_per_job_row);
        auto thread_cols = ceil_div(output_width, xs_per_job_col);
        auto get_thread_index = [&](JobDescription descr) {
                return (get_output_y(descr) / ys_per_job_row) * thread_cols 
                    + get_output_x(descr) / xs_per_job_col;
            };

        std::vector<std::map<JobDescription, op_data_t>> jobs_per_thread;
        jobs_per_thread.resize(thread_rows * thread_cols);

        auto num_of_jobs = 0u;
        for (auto b = 0u; b < batch_iterations; ++b)
        {
            for(auto y = 0u; y < output_height; ++y)
            {
                for(auto x = 0u; x < output_width; ++x)
                {
                    for(auto ky = 0u; ky < filter_height; ++ky)
                    {
                        auto sy  = int32_t(input_begin_y + y * stride_height + ky) - int32_t(filter_offset_y);
                        if ((sy < 0) or (static_cast<uint64_t>(sy) >= full_input_height))
                            continue;

                        for (auto kx = 0u; kx < filter_width; ++kx)
                        {
                            int32_t sx  = int32_t(input_begin_x + x * stride_width + kx) - int32_t(filter_offset_x);
                            if ((sx < 0) or (static_cast<uint64_t>(sx) >= full_input_width))
                                continue;

                            auto index = make_job_descr(b, y, x, sy, sx);
                            auto thread_index = get_thread_index(index);

                            auto curr_input = get_input_block(b, sx, sy);
                            auto curr_weights = get_filter_block(kx, ky);
                            auto curr_output = get_output_block(b, x, y);
                            jobs_per_thread[thread_index][index] = {
                                   curr_output,
                                   curr_input,
                                   curr_weights,
                                   bias,
                                   1
                                };
                            ++num_of_jobs;
                        }
                    }
                }
            }
        }

        {
            std::vector<std::map<JobDescription, op_data_t>> nonempty_jobs_per_thread;
            for (auto& jobs : jobs_per_thread)
                if (jobs.size() > 0)
                    nonempty_jobs_per_thread.push_back(jobs);

            jobs_per_thread.swap(nonempty_jobs_per_thread);
        }

        op_data.resize(jobs_per_thread.size());
        op_array.resize(jobs_per_thread.size());
        jobs.resize(1);
        jobs[0].resize(op_array.size());
        for (auto i = 0u; i < jobs[0].size(); ++i)
        {
            {
                std::set<std::tuple<uint64_t, uint64_t, uint64_t>> outputs;
                for (auto& job : jobs_per_thread[i])
                {
                    auto output = std::make_tuple(
                            get_batch(job.first), get_output_y(job.first), get_output_x(job.first));
                    if (outputs.insert(output).second)
                        job.second.type = 0;
                }
            }
            if (apply_relu)
            {
                std::set<std::tuple<uint64_t, uint64_t, uint64_t>> outputs;
                for (auto& job : reverse(jobs_per_thread[i]))
                {
                    auto output = std::make_tuple(
                            get_batch(job.first), get_output_y(job.first), get_output_x(job.first));
                    if (outputs.insert(output).second)
                        job.second.type = 2;
                }
            }

            for (auto& job : jobs_per_thread[i])
                op_data[i].push_back(job.second);
            op_array[i] = {op_data[i].size(), &op_data[i].front()};
            jobs[0][i] = {reinterpret_cast<void(*)(void*)>(code.getCode()), &op_array[i]};
        }
    }

    std::vector<float*> aux_buffer;

    jit_code code_bias;

    std::vector<detail::FinishJob> finish_data;

    //fully connected version
    jit_convolution_generic(
        uint64_t batch_iterations,
        bool apply_relu,
        uint64_t num_of_threads,
        float*   output, uint64_t output_feature_maps,
        float*   input, uint64_t input_feature_maps,
        float *  filter,
        float*   bias,
        uint64_t input_feats_in_block,
        uint64_t output_feats_in_block)
            : code(input_feats_in_block, output_feats_in_block,
                false, true, false, false, true)
            , code_bias(input_feats_in_block, output_feats_in_block,
                true, true, false, false, false)
    {
        if (output_feature_maps % output_feats_in_block != 0)
            throw std::runtime_error("output_feature_maps % output_feats_in_block != 0");
        if (input_feature_maps % input_feats_in_block != 0)
            throw std::runtime_error("input_feature_maps % input_feats_in_block != 0");

        const auto outfeats_blocks = output_feature_maps / output_feats_in_block;
        const auto infeats_blocks = input_feature_maps / input_feats_in_block;

        {
            std::vector<float> converted_weights(output_feature_maps * input_feature_maps);        
            for (auto iblock = 0u; iblock < infeats_blocks; ++iblock)
            {
                for (auto oblock = 0u; oblock < outfeats_blocks; ++oblock)
                {
                    auto dest_offset = (iblock * outfeats_blocks + oblock)
                                * output_feats_in_block * input_feats_in_block;

                    auto curr_dest_block = &converted_weights.front() + dest_offset;
                    for (auto i = 0u; i < input_feats_in_block; ++i)
                    {
                        for (auto o = 0u; o < output_feats_in_block; ++o)
                        {
                            auto total_o = oblock * output_feats_in_block + o;
                            auto total_i = iblock * input_feats_in_block + i;
                            auto total_oblock = total_o / output_features_per_iteration;
                            auto total_o_in_block = total_o % output_features_per_iteration;

                            auto src = filter[
                                (total_oblock * input_feature_maps + total_i)
                                    * output_features_per_iteration + total_o_in_block];
                            
                            auto oblock = o / output_features_per_iteration;
                            auto o_in_block = o % output_features_per_iteration;
                            auto dest_block_offset =
                                (oblock * input_feats_in_block + i)
                                    * output_features_per_iteration + o_in_block;
                           
                            curr_dest_block[dest_block_offset] = src;
                        }
                    }
                }
            }
            std::memcpy(filter, &converted_weights.front(), converted_weights.size() * sizeof(float));
        }

        auto threads_cols = 4;
        auto threads_rows = 9;
        if (output_feature_maps == 1000)
            threads_cols = 2;

        const auto outblocks_per_thread = (outfeats_blocks + threads_rows - 1) / threads_rows;
        const auto inblocks_per_thread = (infeats_blocks + threads_cols - 1) / threads_cols;
        
        const auto output_buffer_size =
            output_feature_maps
            * batch_size * batch_iterations * sizeof(float);
        aux_buffer.push_back(output);
        for (auto i = 1u; i < threads_cols; ++i)
            aux_buffer.push_back(
                static_cast<float*>(nn_allocate_aligned(output_buffer_size)));

        op_data.resize(threads_cols * threads_rows);
        for (auto iblock = 0u; iblock < infeats_blocks; ++iblock)
        {
            for (auto oblock = 0u; oblock < outfeats_blocks; ++oblock)
            {
                for (auto b = 0u; b < batch_iterations; ++b)
                {
                    auto infeats_thread_col = iblock / inblocks_per_thread;
                    auto index =
                        oblock / outblocks_per_thread * threads_cols
                        + infeats_thread_col;
                    op_data_t job = {
                        aux_buffer[infeats_thread_col]
                            + (b * outfeats_blocks + oblock) * output_feats_in_block * batch_size,
                        input
                            + b * input_feature_maps * batch_size
                            + iblock * input_feats_in_block * batch_size,
                        filter
                            + (iblock * outfeats_blocks + oblock)
                            * output_feats_in_block * input_feats_in_block,
                        bias + oblock * output_feats_in_block,
                        ((iblock % inblocks_per_thread == 0) ? '\0' : '\1')};

                    op_data[index].push_back(job);
                }
            }
        }
        op_array.resize(op_data.size());
        jobs.resize(2);
        jobs[0].resize(op_array.size());
        for (auto i = 0u; i < op_data.size(); ++i)
        {
            op_array[i] = {op_data[i].size(), &op_data[i].front()};

            auto func = reinterpret_cast<void(*)(void*)>(
                (i % threads_cols == 0) ? code_bias.getCode() : code.getCode());
            jobs[0][i] = {func, &op_array[i] };
        }

        num_of_threads = 6;
        finish_data.resize(batch_iterations * num_of_threads);
        jobs[1].resize(batch_iterations * num_of_threads);
        auto outputs_per_thread =
            (output_feature_maps + num_of_threads - 1) / num_of_threads;

        for (auto b = 0u; b < batch_iterations; ++b)
        {
            for (auto i = 0u; i < num_of_threads; ++i)
            {
                auto job = b * num_of_threads + i;
                auto curr_offset = b * output_feature_maps * batch_size
                    + i * outputs_per_thread * batch_size;
                auto curr_outfeats = (i + 1) * outputs_per_thread;
                if (curr_outfeats > output_feature_maps)
                    curr_outfeats = output_feature_maps;
                curr_outfeats -= i * outputs_per_thread;

                finish_data[job].output_dest = output + curr_offset;
                for (auto j = 1u; j < threads_cols; ++j)
                    finish_data[job].outputs.push_back(aux_buffer[j] + curr_offset);
                finish_data[job].ofeats = curr_outfeats;

                if (apply_relu)
                    jobs[1][job] = { detail::finish_with_relu, &finish_data[job] };
                else
                    jobs[1][job] = { detail::finish_without_relu, &finish_data[job] };
            }
        }
    }
};

