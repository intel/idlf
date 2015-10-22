#include <asmjit/asmjit.h>
#include "nn_asmjit_compilation.h"

namespace nn
{
namespace asm_compilation
{
namespace
{
asmjit::JitRuntime asmjitRuntime;
} //namespace

void* asmjit_compile(Compilation compile)
{
    using namespace asmjit;
    using namespace asmjit::x86;
    FileLogger logger(stdout);
    logger.setOption(kLoggerOptionBinaryForm, true);
    X86Assembler assembler(&asmjitRuntime);
    //assembler.setLogger(&logger);

    assembler.push(rbp);
    assembler.push(rsp);
    assembler.push(rdx);
    assembler.push(rbx);
    assembler.push(r12);
    assembler.push(r13);
    assembler.push(r14);
    assembler.push(r15);

    for (auto i = 6; i < 16; ++i)
    {
        assembler.sub(rsp, 16);
        assembler.movdqu(ptr(rsp), xmm(i));
    }

    compile(assembler);

    for (auto i = 15; i >= 6; --i)
    {
        assembler.movdqu(xmm(i), ptr(rsp));
        assembler.add(rsp, 16);
    }
    assembler.pop(r15);
    assembler.pop(r14);
    assembler.pop(r13);
    assembler.pop(r12);
    assembler.pop(rbx);
    assembler.pop(rdx);
    assembler.pop(rsp);
    assembler.pop(rbp);
    assembler.ret();
    return assembler.make();
}

void asmjit_func_release(void* func_ptr)
{
    asmjitRuntime.release(func_ptr);
}

} //namespace asm_compilation
} //namespace nn
