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
#include "cpu/naive_implementations.h"
#include "cpu/forward_convolve_simmplifying_wrapper.h"
#include "cpu/test_helpers.h"
#include "cpu/core/nn_asmjit_power.h"
#include "cpu/core/nn_intrinsic_power.h"
#include <cmath>

namespace
{

struct Args { float* input; float* output; };
void compile(asmjit::X86Assembler& a)
{
    using namespace asmjit;
    using namespace asmjit::x86;
    auto inputPtrParam = nn_asmjit_param_ptr(Args, input);
    auto outputPtrParam = nn_asmjit_param_ptr(Args, output);

    a.mov(r8, inputPtrParam);
    a.mov(r9, outputPtrParam);
    a.vmovups(ymm0, ptr(r8));

    nn::asm_compilation::generate_invpow075(a, ymm0, ymm1, ymm2, ymm3, ymm4);

    a.vmovups(ptr(r9), ymm0);
}

struct AsmjitModulesTest : ::testing::Test
{
    typedef void (*InvPow075)(Args*);

    InvPow075 func;

    AsmjitModulesTest() : func(reinterpret_cast<InvPow075>(nn::asm_compilation::asmjit_compile(compile))) {}

    void check_intrinsic_pow_075(float* vals) //must be 8
    {
        float output[8];
        nn::intrinsics::invpow_075(vals, output);
        for (auto i = 0u; i < 8; ++i)
            ASSERT_FLOAT_CHECK(std::pow(vals[i], -0.75), output[i]) << "[" << i << "] = " << vals[i] << "^-0.75";
    }
    void check_pow_075(float* vals) //must be 8
    {
        float output_asmjit[8];
        float output_intrinsics[8];
        Args args = {vals, output_asmjit};
        func(&args);
        nn::intrinsics::invpow_075(vals, output_intrinsics);
        for (auto i = 0u; i < 8; ++i)
        {
            ASSERT_FLOAT_CHECK(std::pow(vals[i], -0.75), output_intrinsics[i]) << "intrinsic output[" << i << "] = " << vals[i] << "^-0.75";
            ASSERT_EQ(output_intrinsics[i], output_asmjit[i]);
            ASSERT_FLOAT_CHECK(std::pow(vals[i], -0.75), output_asmjit[i]) << "asmjit output[" << i << "] = " << vals[i] << "^-0.75";
        }
    }

    void check_pow_075(std::vector<float> values)
    {
        ASSERT_EQ(0, values.size() % 8);
        for (auto i = 0u; i < values.size(); i += 8)
            ASSERT_NO_FATAL_FAILURE(check_pow_075(&values.front() + i)) << "batch " << i;
    }
};

TEST_F(AsmjitModulesTest, power075_specific_values)
{
    auto inv = 4.0f/3.0f;
    check_pow_075({0.00000000001f, 1.0f, 1.0f, 2.0f, 8.0f, 16.0f, 3.0f, 5.0f,
                   3.14159265358979f,
                   static_cast<float>(exp(1.0)),
                   static_cast<float>(pow(2.0f, inv)),
                   static_cast<float>(pow(7.123, inv)),
                   static_cast<float>(pow(0.0000000001, inv)),
                   std::numeric_limits<float>::min(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max() / 2.0f});
}

TEST_F(AsmjitModulesTest, power075_big_values)
{
    std::vector<float> some_values(1024);
    randomizeWithAny(some_values.begin(), some_values.end(),
        0, std::numeric_limits<float>::max());
    check_pow_075(some_values);
}

TEST_F(AsmjitModulesTest, power075_small_values)
{
    std::vector<float> some_values(1024);
    randomizeWithAny(some_values.begin(), some_values.end(), 0.0, 1.0);
    check_pow_075(some_values);
}

} //namespace

