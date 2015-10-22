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

#include "device/common/nn_asmjit_compilation.h"

namespace nn
{
namespace asm_compilation
{
//domain is numbers greater than 0 (for negative it is undefined, for 0 it is large value, but not infinity)
template <typename YmmReg>
void generate_invpow075(asmjit::X86Assembler& a,
                        YmmReg arg,
                        YmmReg aux1,
                        YmmReg aux2,
                        YmmReg aux3,
                        YmmReg aux4)
{
    using namespace asmjit;
    using namespace asmjit::x86;

    auto& mantissa = aux3;
    broadcast_imm(a, mantissa, 0x007fffff);
    a.vpand(aux2, arg, mantissa);
    broadcast_imm(a, aux1, 0x3f800000);
    a.vpor(mantissa, aux2, aux1);
    //still important: arg, aux3

    static float coeff1 = -0.06251362156237f;
    static float coeff2 = 0.56657226995864f;
    static float coeff3 = -2.12314847503624f;
    static float coeff4 = 4.22879355263332f;
    static float coeff5 = -4.79039952143706f;
    static float coeff6 = 3.18069569544757f;

    std::vector<float> coeffs = {
        -2.12314847503624f, 4.22879355263332f, -4.79039952143706f, 3.18069569544757f};

    auto& mantissa_result = aux1;
    broadcast_imm(a, mantissa_result, 0.56657226995864f);
    broadcast_imm(a, aux2, -0.06251362156237f);
    a.vfmadd231ps(mantissa_result, mantissa, aux2);
    for (auto& coeff : coeffs)
    {
        broadcast_imm(a, aux2, coeff);
        a.vfmadd213ps(mantissa_result, mantissa, aux2);
    }
    //still important: arg, aux1


    auto& e = aux2;
    broadcast_imm(a, aux3, 0x7f800000);
    a.vpand(e, arg, aux3);
    broadcast_imm(a, aux3, 0x3f800000);
    a.vpsubw(e, e, aux3);
    a.vpslld(e, e, 1);
    //still important: aux1, aux2

    auto& p2 = aux4;
    auto& partial_result = mantissa_result; //aux1
    broadcast_imm(a, aux3, 2 << 24);
    a.vpand(p2, e, aux3);
    broadcast_imm(a, aux3, 0);
    a.vpcmpeqd(p2, p2, aux3);
    broadcast_imm(a, aux3, 1.0f);
    broadcast_imm(a, arg, 0.35355339059327376220042218105f);
    a.vblendvps(p2, arg, aux3, p2);
    a.vmulps(partial_result, partial_result, p2);
    ////still important: aux1, aux2

    auto& p0 = arg;
    broadcast_imm(a, aux3, 0xfc000000);
    a.vpand(p0, e, aux3);
    a.vpsrad(p0, p0, 2);
    broadcast_imm(a, aux3, -3);
    a.vpmulld(p0, p0, aux3);
    broadcast_imm(a, aux3, 0x7f000000);
    a.vpaddd(p0, p0, aux3);
    a.vpsrlq(p0, p0, 1);
    //still important: aux1, aux2, arg

    auto& p1 = aux4;
    broadcast_imm(a, aux3, 1 << 24);
    a.vpand(p1, e, aux3);
    broadcast_imm(a, aux3, 0);
    a.vpcmpeqd(p1, p1, aux3);
    broadcast_imm(a, aux3, 1.0f);
    broadcast_imm(a, e, 0.59460355750136053335874998528f);
    a.vblendvps(p1, e, aux3, p1);
    a.vmulps(arg, p0, p1);
    //still important: arg, aux1

    a.vmulps(arg, arg, partial_result);
}
} //namespace asm_compilation
} //namespace nn

