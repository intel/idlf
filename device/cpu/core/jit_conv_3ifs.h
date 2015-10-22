#pragma once

#include <functional>
#include <set>
#include <map>
#include <random>
#include <iostream>
#include <algorithm>
#include "convolution_jit.h"
#include "device/common/nn_allocate.h"
#include "xbyak/xbyak.h"
#undef max
#undef min
#include <cstddef>
#include <cstdlib>

// job set of jit convolutions ////////////////////////////////////////////////

struct jit_convolution_zxyn : public jit_convolution
{
#pragma pack(push, 1)
    struct op_data_t
    {
        float *output;
        uint64_t input_offset;
        float *filter;
        float *bias;
        float *input;
    };
#pragma pack(pop)

    static const uint64_t output_features_per_iteration = 16;
    static const uint64_t register_width_in_float       = 8;
    static const uint64_t register_width                = register_width_in_float * sizeof(float);

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

        jit_code(uint64_t output_width,
                 uint64_t output_height,
                 uint64_t output_feats,
                 uint64_t input_width,
                 uint64_t input_feats,
                 uint64_t filter_height,
                 uint64_t filter_width,
                 uint64_t stride_x,
                 uint64_t stride_y,
                 bool apply_relu,
                 void* code_ptr = nullptr,
                 size_t code_size = 40 * Xbyak::DEFAULT_MAX_CODE_SIZE)
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

            auto aux_input                = nn_jit_dangerous_reg1;
            auto aux_input_outer          = nn_jit_dangerous_reg2;
            auto aux_output               = r9;
            auto aux_filter               = r15;

            auto left_out_rows            = r8;
            auto left_out_blocks          = r14;
            auto left_kern_rows           = rdx;

            auto generate_for_single_output_block = [&](std::string tag,
                                                        uint64_t output_blocks,
                                                        bool vertical) {

                    const auto accumulators_count   = output_blocks * 2;

                    align(4);
                    mov(aux_input, aux_input_outer);
                    mov(aux_filter, filter);
                    for (int n = 0; n < accumulators_count; ++n)
                        vxorps(Ymm(n), Ymm(n), Ymm(n));

                    mov(left_kern_rows, filter_height);
                    L(tag + "_kern_row");
                    {
                        for (auto kern_col = 0u; kern_col < filter_width; ++kern_col)
                        {
                            for (auto i = 0u; i < input_feats; ++i)
                            {
                                auto filter_offset =
                                    (kern_col * input_feats * output_features_per_iteration
                                    + i * output_features_per_iteration)
                                        * sizeof(float);
                                vmovaps(ymm13, ptr [aux_filter + filter_offset]);
                                vmovaps(ymm14, ptr [aux_filter + filter_offset + register_width]);

                                for (auto j = 0u; j < output_blocks; ++j)
                                {
                                    auto block_offset = j * stride_x;
                                    if (vertical) block_offset = j * stride_y * input_width;
                                    auto input_offset =
                                        (block_offset * input_feats
                                        + kern_col * input_feats
                                        + i) * sizeof(float);
                                    vbroadcastss(ymm15, ptr [aux_input + input_offset]);
                                    vfmadd231ps(Ymm(2 * j + 0), ymm13, ymm15);
                                    vfmadd231ps(Ymm(2 * j + 1), ymm14, ymm15);
                                }
                            }
                        }
                        add(aux_input, input_width * input_feats * sizeof(float));
                        add(aux_filter, filter_width * input_feats * output_features_per_iteration * sizeof(float));
                    }
                    dec(left_kern_rows);
                    jnz(tag + "_kern_row");

                    for (int n = 0; n < accumulators_count; ++n)
                        vaddps(Ymm(n), ptr [bias + (n % 2) * register_width]);
                    if (apply_relu)
                    {
                        vxorps(ymm15, ymm15, ymm15);
                        for (int n = 0; n < accumulators_count; ++n)
                            vmaxps(Ymm(n), Ymm(n), ymm15);
                    }
                    for (int o = 0; o < output_blocks; ++o)
                    {
                        auto output_offset = o * output_feats * sizeof(float);
                        if (vertical) output_offset *= output_width;
                        vmovaps(ptr[aux_output + output_offset], Ymm(2 * o));
                        vmovaps(ptr[aux_output + output_offset + register_width], Ymm(2 * o + 1));
                    }
                };


            auto prepare_input_ptr = [&]{
                mov(input,  ptr [nn_jit_param_reg + offsetof(op_data_t, input)]);
                add(input,  ptr [nn_jit_param_reg + offsetof(op_data_t, input_offset)]);
            };

            mov(output, ptr [nn_jit_param_reg + offsetof(op_data_t, output)]);
            mov(filter, ptr [nn_jit_param_reg + offsetof(op_data_t, filter)]);
            mov(bias,   ptr [nn_jit_param_reg + offsetof(op_data_t, bias)]);
            prepare_input_ptr();

            mov(left_out_rows, output_height);
            L("output_row");
            {
                mov(aux_input_outer, input);
                mov(aux_output, output);

                mov(left_out_blocks, output_width / 6);
                L("output_block");
                {
                    generate_for_single_output_block("horizontal_full", 6, false);

                    add(aux_output, 6 * output_feats * sizeof(float));
                    add(aux_input_outer, 6 * input_feats * stride_x * sizeof(float));
                }
                dec(left_out_blocks);
                jnz("output_block");

                add(input, input_feats * input_width * stride_y * sizeof(float));
                add(output, output_feats * output_width * sizeof(float));

            }
            dec(left_out_rows);
            jnz("output_row");

            mov(output, ptr [nn_jit_param_reg + offsetof(op_data_t, output)]);
            prepare_input_ptr();
            for (auto i = 0u; i < output_height / 6; ++i)
            {
                mov(aux_input_outer, input);
                mov(aux_output, output);
                add(aux_output, (i * output_width + output_width / 6) * 6 * output_feats * sizeof(float));
                add(aux_input_outer, (i * input_width * stride_y + output_width / 6 * stride_x) * 6 * input_feats * sizeof(float));
                for (auto j = 0u; j < output_height % 6; ++j)
                {
                    generate_for_single_output_block("vertical_full_" + std::to_string(i), 6, true);
                    add(aux_output, output_feats * sizeof(float));
                }
            }
            for (auto i = output_height / 6 * 6; i < output_height; ++i)
            {
                mov(aux_input_outer, input);
                mov(aux_output, output);
                add(aux_output, (i * output_width + output_width / 6 * 6) * output_feats * sizeof(float));
                add(aux_input_outer, (i * input_width * stride_y + output_width / 6 * stride_x * 6) * input_feats * sizeof(float));
                generate_for_single_output_block("horizontal_partial", output_width % 6, false);
            }

            postamble();
        }
    };

    std::vector<op_data_t> op_data;
    jit_code code;

    jit_convolution_zxyn(
        uint64_t batch,
        bool apply_relu,
        
        float*   output,
        uint64_t output_width,
        uint64_t output_height,
        uint64_t output_feature_maps,

        uint64_t input_width,
        uint64_t input_height,
        uint64_t input_feature_maps,

        uint64_t stride_width,
        uint64_t stride_height,

        float *  filter,
        uint64_t filter_width,
        uint64_t filter_height,
        
        float*   bias)
            : code(output_width, output_height, output_feature_maps,
                   input_width, input_feature_maps,
                   filter_height, filter_width,
                   stride_width, stride_height,
                   apply_relu)
    {
        assert(output_feature_maps % output_features_per_iteration == 0);

        auto z_out_blocks = output_feature_maps / output_features_per_iteration;
        const auto batch_block_size = 24;
        for (auto bblock = 0u; bblock < batch / batch_block_size; ++bblock)
        {
            for (auto z_block = 0u; z_block < z_out_blocks; ++z_block)
            {
                for (auto b = bblock * batch_block_size; b < (bblock + 1) * batch_block_size; ++b)
                {
                    auto output_shifted = output
                        + b * output_width * output_height * output_feature_maps
                        + z_block * output_features_per_iteration;
                    auto bias_shifted = bias + z_block * output_features_per_iteration;
                    auto filter_shifted = filter
                        + z_block * output_features_per_iteration * input_feature_maps * filter_width * filter_height;
                    auto input_shifted = b * input_width * input_height * input_feature_maps;

                    op_data.push_back({ output_shifted,
                        input_shifted * sizeof(float),
                        filter_shifted,
                        bias_shifted });
                }
            }
        }
        for (auto z_block = 0u; z_block < z_out_blocks; ++z_block)
        {
            for (auto b = batch / batch_block_size * batch_block_size; b < batch; ++b)
            {
                auto output_shifted = output
                    + b * output_width * output_height * output_feature_maps
                    + z_block * output_features_per_iteration;
                auto bias_shifted = bias + z_block * output_features_per_iteration;
                auto filter_shifted = filter
                    + z_block * output_features_per_iteration * input_feature_maps * filter_width * filter_height;
                auto input_shifted = b * input_width * input_height * input_feature_maps;

                op_data.push_back({ output_shifted,
                    input_shifted * sizeof(float),
                    filter_shifted,
                    bias_shifted });
            }
        }

        jobs.resize(1);
        jobs[0].resize(op_data.size());
        for (auto i = 0u; i < jobs[0].size(); ++i)
            jobs[0][i] = {reinterpret_cast<void(*)(void*)>(code.getCode()), &op_data[i]};
    }
};

