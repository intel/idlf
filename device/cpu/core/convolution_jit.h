#pragma once

#include <vector>


#ifdef __linux__
#define nn_jit_param_reg rdi
#define nn_jit_dangerous_reg1 rcx
#define nn_jit_dangerous_reg2 rbx
#define nn_jit_dangerous_reg3 rsi
#define nn_jit_dangerous_reg4 rbp
#define nn_jit_dangerous_reg5 rsp
#else
#define nn_jit_param_reg rcx
#define nn_jit_dangerous_reg1 rdi
#define nn_jit_dangerous_reg2 rbx
#define nn_jit_dangerous_reg3 rsi
#define nn_jit_dangerous_reg4 rbp
#define nn_jit_dangerous_reg5 rsp
#endif

// interface for convolution variant using JIT
struct jit_convolution
{
    std::vector<std::vector<nn_multithreaded_request>> jobs;

    virtual ~jit_convolution() {};
};

template<typename T> class reverse_t
{
    T& ref;
public:
    reverse_t(T& arg): ref(arg) {}
    auto begin() const -> decltype(ref.rbegin()) { return ref.rbegin (); }
    auto end()   const -> decltype(ref.rend())   { return ref.rend (); }
};
template<typename T> reverse_t<const T> reverse(const T& x) { return reverse_t<const T>(x); }
template<typename T> reverse_t<T> reverse(T& x) { return reverse_t<T>(x); }
