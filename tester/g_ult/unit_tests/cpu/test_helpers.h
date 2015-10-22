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

#include <random>
#include <thread>
#include <cassert>

inline uint64_t ceil_div(uint64_t arg, uint64_t div)
{
    return (arg + div - 1) / div;
}

template <typename Cont>
void randomize(Cont& dest, const Cont& vals)
{
    static std::mt19937 e1(11);
    std::uniform_int_distribution<int> ud(0, vals.size() - 1);

    for (auto& elem : dest)
    {
        auto index = ud(e1);
        assert(index >= 0);
        assert(index < vals.size());
        elem = vals[index];
    }
}

template <typename It>
void randomizeWithAny(It begin, It end, float from = 1.0, float till = 10.0, uint32_t seed = 123)
{
    std::mt19937 e1(123);
    std::uniform_real_distribution<float> ud(from, till);

    for (It it = begin; it != end; ++it)
        *it = ud(e1);

    
}

template <typename It>
void randomizeWithAnyThreaded(It begin, It end, uint32_t threads)
{
    auto size = (end - begin);
    if (size < threads) return randomizeWithAny(begin, end);
    size /= threads;
    std::vector<std::thread> futs(threads);
    for (uint64_t i = 0; i < threads; ++i)
    {
        futs[i] = std::thread(
            [=]{ randomizeWithAny(begin + i * size, begin + (i + 1) * size); });
            //randomizeWithAny(begin + i * size, begin + (i + 1) * size);
    }
    randomizeWithAny(begin + size * threads, end);
    for (auto& fut : futs) fut.join();
    for (auto it = begin; it != end; ++it)
        assert(*it != 0);
}

inline bool float_check(float ref, float val, float error = 0.0002)
{
    auto diff = std::fabs(ref - val);
    if (ref == 0) return (std::fabs(val) < error);
    if (std::fabs(ref) < 1.0)
        return (diff < std::fabs(ref * pow(error, ref)));

    return (diff < std::fabs(ref * error));
}

#define ASSERT_FLOAT_CHECK(left, right) ASSERT_TRUE(float_check(left, right)) \
	<< left << " != " << right << " by margin " << std::fabs((float)left - (float)right) << " "
#define EXPECT_FLOAT_CHECK(left, right) EXPECT_TRUE(float_check(left, right)) \
	<< left << " != " << right << " by margin " << std::fabs((float)left - (float)right) << " "


