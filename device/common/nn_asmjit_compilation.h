#pragma once

#include <functional>
#include <cstddef>
#include <unordered_set>
#include <asmjit/asmjit.h>

#ifdef __linux__
#define nn_asmjit_param_reg asmjit::x86::rdi
#define nn_asmjit_dangerous_reg1 asmjit::x86::rcx
#define nn_asmjit_dangerous_reg2 asmjit::x86::rbx
#define nn_asmjit_dangerous_reg3 asmjit::x86::rsi
#define nn_asmjit_dangerous_reg4 asmjit::x86::rbp
#define nn_asmjit_dangerous_reg5 asmjit::x86::rsp
#else
#define nn_asmjit_param_reg asmjit::x86::rcx
#define nn_asmjit_dangerous_reg1 asmjit::x86::rdi
#define nn_asmjit_dangerous_reg2 asmjit::x86::rbx
#define nn_asmjit_dangerous_reg3 asmjit::x86::rsi
#define nn_asmjit_dangerous_reg4 asmjit::x86::rbp
#define nn_asmjit_dangerous_reg5 asmjit::x86::rsp
#endif

#define nn_asmjit_ptr(reg, type, field) \
    asmjit::x86::ptr(reg, offsetof(type, field), sizeof(((type*)nullptr)->*(&type::field)))
#define nn_asmjit_param_ptr(type, field) \
    nn_asmjit_ptr(nn_asmjit_param_reg, type, field)

namespace asmjit
{
struct X86Assembler;
} //asmjit

namespace nn
{
namespace asm_compilation
{
typedef std::function<void(asmjit::X86Assembler&)> Compilation;

void* asmjit_compile(Compilation);
void asmjit_func_release(void* func_ptr);

template <typename CompiledFuncType>
void release(CompiledFuncType* func_ptr)
{
    asmjit_func_release((void*)func_ptr);
}

typedef std::function<void(asmjit::X86Assembler& a)> CalcFunc;

template <typename It, typename Start, typename Till>
CalcFunc loop(It it, Start from, Till till, CalcFunc internal)
{
    using namespace asmjit;
    /*
       for (auto it = from; it < till; ++it)
           internal();
    */
    return [=](X86Assembler& a){
            Label loop_exit(a);
            Label loop_enter(a);

            a.mov(it, from);

            a.bind(loop_enter);
            a.cmp(it, till);
            a.jae(loop_exit);

            internal(a);

            a.inc(it);
            a.jmp(loop_enter);
            a.bind(loop_exit);
        };
}

template <typename YmmReg, typename T>
void broadcast_imm(asmjit::X86Assembler& a, YmmReg& dest, T value)
{
    using namespace asmjit;
    using namespace asmjit::x86;
    static std::unordered_set<T> aux;
    auto& elem = *aux.insert(value).first;
    a.mov(rax, (size_t)(&elem));
    a.vbroadcastss(dest, ptr(rax));
}

} //namespace asm_compilation
} //namespace nn

