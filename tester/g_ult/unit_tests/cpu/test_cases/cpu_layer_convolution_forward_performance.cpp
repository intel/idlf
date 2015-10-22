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
#include "cpu/core/layer_convolution_avx2_forward.h"
#include "common/nn_allocate.h"
#include "cpu/naive_implementations.h"
#include "cpu/test_helpers.h"
#include <cmath>
#include <chrono>
#include <thread>

#ifndef CORES
#define CORES 18
#endif

namespace
{
using namespace convolution::forward;
using namespace convolution;

uint64_t calcFmas(InputDimensions in_dims, OutputDimensions out_dims, KernelInfo kern_info, uint64_t batch)
{
    uint64_t ret = 0u;
    for (auto ofeat = 0u; ofeat < out_dims.feats; ++ofeat)
    {
        for (auto row = 0u; row < out_dims.height; ++row)
        {
            for (auto col = 0u; col < out_dims.width; ++col)
            {
                for (auto krow = 0u; krow < kern_info.dims.height; ++krow)
                {
                    if (row + krow < kern_info.center.row) continue;
                    auto irow = row + krow - kern_info.center.row;
                    if (irow >= in_dims.height) continue;

                    for (auto kcol = 0u; kcol < kern_info.dims.width; ++kcol)
                    {
                        if (col + kcol < kern_info.center.col) continue;
                        auto icol = col + kcol - kern_info.center.col;
                        if (icol >= in_dims.width) continue;

                        ret += in_dims.feats * (batch / 8);
                    }
                }
            }
        }
    }
    return ret;
}

std::vector<float> transposeFilter(
    float* filter,
    KernelDimensions kern_dims,
    ValueU64<InputFeats> ifeats,
    ValueU64<OutputFeats> ofeats,
    uint64_t ofeat_block_size)
{
    std::vector<float> ret(kern_dims.size() * ifeats * ofeats, 0);
    uint64_t ofeat_blocks = ofeats / ofeat_block_size;
    for (uint64_t row = 0u; row < kern_dims.height; ++row)
        for (uint64_t col = 0u; col < kern_dims.width; ++col)
            for (uint64_t ofeat_block = 0u; ofeat_block < ofeat_blocks; ++ofeat_block)
                for (uint64_t ifeat = 0u; ifeat < ifeats; ++ifeat)
                    for (uint64_t ofeat_in_block = 0u; ofeat_in_block < ofeat_block_size; ++ofeat_in_block)
                    {
                        auto ofeat = ofeat_block_size * ofeat_block + ofeat_in_block;
                        ret[((ofeat * kern_dims.height + row) * kern_dims.width + col) * ifeats + ifeat]
                            = filter[(((row * kern_dims.width + col) * ofeat_blocks + ofeat_block) * ifeats + ifeat)
                                * ofeat_block_size + ofeat_in_block];
                    }
    return ret;
}

std::vector<float> extractInput(
    float* input, InputDimensions in_dims, uint64_t batch, uint64_t batch_block_size)
{
    std::vector<float> ret(in_dims.size(), 0);
    for (uint64_t row = 0u; row < in_dims.height; ++row)
    {
        for (uint64_t col = 0u; col < in_dims.width; ++col)
        {
            for (uint64_t ifeat = 0u; ifeat < in_dims.feats; ++ifeat)
            {
                ret[(row * in_dims.width + col) * in_dims.feats + ifeat]
                    = input[((row * in_dims.width + col) * in_dims.feats + ifeat) * batch_block_size + batch];
            }
        }
    }
    return ret;
}

struct BatchBlock {};
struct BatchIndex {};
std::vector<float> extractOutput(
    float* output, OutputDimensions out_dims, ValueU64<BatchIndex> batch, ValueU64<BatchBlock> batch_block_size)
{
    std::vector<float> ret(out_dims.size(), 0);
    for (uint64_t row = 0u; row < out_dims.height; ++row)
    {
        for (uint64_t col = 0u; col < out_dims.width; ++col)
        {
            for (uint64_t ofeat = 0u; ofeat < out_dims.feats; ++ofeat)
            {
                ret[(ofeat * out_dims.height + row) * out_dims.width + col] =
                    output[((row * out_dims.width + col) * out_dims.feats + ofeat) * batch_block_size + batch];
            }
        }
    }
    return ret;
}

struct OutFeatBlockSize {};
void checkValues(float* input, float* output, float* filter,
    InputDimensions in_dims, OutputDimensions out_dims, KernelInfo kern_info,
    ValueU64<Batch> batch, ValueU64<BatchBlock> batchBlock, ValueU64<OutFeatBlockSize> ofeat_block_size)
{                                                 
    std::vector<float> empty_bias(out_dims.feats, 0);
    ASSERT_TRUE(out_dims.feats % ofeat_block_size == 0);
    auto transposedFilter = transposeFilter(filter, kern_info.dims, in_dims.feats, out_dims.feats, ofeat_block_size);
    for(uint64_t i = 0u; i < batch / batchBlock; ++i)
    {
        std::cerr << "processing batch " << i << "/" << (batch / batchBlock) << std::endl;
        std::vector<bool> results(batchBlock, true);
        std::vector<std::thread> threads(batchBlock);
        for(uint64_t b = 0u; b < batchBlock; ++b)
        {
            threads[b] = std::thread([=, &transposedFilter, &results]{
                auto singleInput = extractInput(input + i * (batchBlock * in_dims.size()), in_dims, b, batchBlock);
                auto singleOutput = extractOutput(output + i * (batchBlock * out_dims.size()), out_dims, make<BatchIndex>(b), batchBlock);
                auto refOutput = decltype(singleOutput)(singleOutput.size(), 0);

                ult_nn_naive_convolve(
                    &singleInput.front(), &refOutput.front(), &transposedFilter.front(), (float*)&empty_bias.front(),
                    out_dims.feats, in_dims.feats, out_dims.width, out_dims.height, in_dims.width, in_dims.height,
                    kern_info.dims.width, kern_info.dims.height, 1, 1, kern_info.center.col, kern_info.center.row);
                for (uint64_t j = 0u; j < out_dims.size(); ++j)
                    results[b] = results[b] and float_check(refOutput[j], singleOutput[j], 0.0001);
                    //ASSERT_TRUE(float_check(refOutput[j], singleOutput[j], 0.0001))
                        //<< j << " ref: " << refOutput[j] << ", actual: " << singleOutput[j];
            });
        }
        for (auto& thread : threads)
            thread.join();
        for(uint64_t b = 0u; b < batchBlock; ++b)
            ASSERT_TRUE(results[b]) << " batch no: " << i << "/" << (batch / batchBlock)
                                    << " and " << b << "/" << batchBlock;
    }
}

struct PerfTestSizes
{
    InputDimensions in_dims;
    KernelDimensions kern_size;
    ValueU64<OutputFeats> out_feats;
    uint64_t loops;

    PerfTestSizes(InputDimensions in_dims,
                  KernelDimensions kern_size,
                  ValueU64<OutputFeats> out_feats,
                  uint64_t loops)
    : in_dims(in_dims)
    , kern_size(kern_size)
    , out_feats(out_feats)
    , loops(loops)
    {}
};

struct PerfTest : ::testing::TestWithParam<PerfTestSizes> {};

TEST_P(PerfTest, DISABLED_performance)
{
    uint64_t threads = CORES * 2;
    nn_thread_worker_pool thread_pool(threads);
    uint64_t cores = CORES;
    auto in_dims = GetParam().in_dims;
    OutputDimensions out_dims(
        make<OutputHeight>(in_dims.height),
        make<OutputWidth>(in_dims.width),
        GetParam().out_feats);

    KernelCenter kern_center = {
        make<Rows>(GetParam().kern_size.height / 2),
        make<Cols>(GetParam().kern_size.width / 2)};
    KernelInfo kern_info = {GetParam().kern_size,
        GetParam().out_feats, kern_center, Stride{make<Rows>(1), make<Cols>(1)}};
    StrideInfo stride = {kern_info.stride, InputStart{make<Rows>(0), make<Cols>(0)}};
    uint64_t batch = BATCH_ACCEPTED_BLOCK;
    uint64_t loops = GetParam().loops;
    uint64_t tests = 2u;//1000u;

    ASSERT_EQ(0, in_dims.feats % INPUT_ACCEPTED_MOD);
    ASSERT_EQ(0, out_dims.feats % OUTPUT_ACCEPTED_MOD);
    ASSERT_EQ(0, batch % BATCH_ACCEPTED_BLOCK);

    std::cerr <<
        "width: " << in_dims.width << " height: " << in_dims.height << std::endl <<
        "kernel width: " << kern_info.dims.width << " kernel height: " << kern_info.dims.height << std::endl <<
        "input features: " << in_dims.feats << " output features: " << out_dims.feats << std::endl <<
        "batch block : " << batch << " loops: " << loops << std::endl;

    uint64_t inputSize = in_dims.size() * batch;
    uint64_t outputSize = out_dims.size() * batch;
    uint64_t weightsSize = kern_info.dims.size() * out_dims.feats;

    auto alignedInput = nn_make_unique_aligned<float>(inputSize * loops);
    auto alignedWeights = nn_make_unique_aligned<float>(weightsSize);
    auto alignedOutput = nn_make_unique_aligned<float>(outputSize * loops);
    std::vector<float> bias(out_dims.feats, 0);

    randomizeWithAnyThreaded(alignedInput.get(), alignedInput.get() + inputSize * loops, 2 * CORES);
    randomizeWithAnyThreaded(alignedWeights.get(), alignedWeights.get() + weightsSize, 2 * CORES);
    //randomizeWithAnyThreaded(bias.begin(), bias.end(), 2 * CORES);
    auto convInternal = compileConvolution(
        threads,
        kern_info,
        in_dims,
        in_dims,
        out_dims,
        out_dims,
        stride,
        make<InputFeatsStart>(0u),
        make<OutputFeatsStart>(0u),
        (batch + BATCH_ACCEPTED_BLOCK - 1) / BATCH_ACCEPTED_BLOCK,
        alignedInput.get(),
        alignedOutput.get(),
        alignedWeights.get(),
        &bias.front());

    uint64_t min_time = UINT64_MAX;
    uint64_t max_time = 0;
    uint64_t sum_time = 0;
    uint64_t fmas = calcFmas(in_dims, out_dims, kern_info, batch);
    uint64_t zero_fmas = (uint64_t)in_dims.size() * (uint64_t)kern_info.dims.size() / kern_info.dims.feats * (uint64_t)out_dims.feats * (uint64_t)(batch / 8u);
    std::cerr << "fmas per execution: " << fmas << std::endl;
    for(auto l = 0u; l < tests; ++l)
    {
        memset(alignedOutput.get(), 0, sizeof(float) * outputSize * loops);
        std::cerr << "iteration " << l << std::endl;
        uint64_t time = __rdtsc();
        auto start = std::chrono::system_clock::now();

        convolve_fst_threaded_batch(thread_pool, convInternal,
            alignedInput.get(), alignedOutput.get(), alignedWeights.get(), &bias.front(),
            make<Batch>(loops * batch), kern_info, in_dims, out_dims, stride,
            make<InputFeatsStart>(0), make<OutputFeatsStart>(0));

        time = (__rdtsc() - time) * cores / loops;
        auto end = std::chrono::system_clock::now();
        uint64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        uint64_t cycles = (elapsed * 2400u) * cores / loops;

        std::cerr << "rdtsc: " << time << " chrono:" << cycles << std::endl;

        min_time = std::min(time, min_time);
        max_time = std::max(time, max_time);
        sum_time += time; 
        double average_cycles = (double)sum_time / (double)(l + 1);
        std::cerr << "cycles: " <<  time
                  << "\t\t min:" << min_time
                  << "\t\t max:" << max_time
                  << "\t average: " << average_cycles << std::endl;
        std::cerr << "fmas per cycle: " << ((double)fmas / (double)time)
            << "\t min:" << ((double)fmas / (double)max_time)
            << "\t max:" << ((double)fmas / (double)min_time)
            << "\t ave:" << ((double)fmas / average_cycles)
            << std::endl;
        std::cerr << "fmas saturation: " << ((double)fmas / (double)time) / 2.0
            << "\t min:" << ((double)fmas / (double)max_time) / 2.0
            << "\t max:" << ((double)fmas / (double)min_time) / 2.0
            << "\t ave:" << ((double)fmas / average_cycles) / 2.0
            << std::endl;
        std::cerr << "zero fmas per cycle: " << ((double)zero_fmas / (double)time)
            << "\t min:" << ((double)zero_fmas / (double)max_time)
            << "\t max:" << ((double)zero_fmas / (double)min_time)
            << "\t ave:" << ((double)zero_fmas / average_cycles)
            << std::endl;
        std::cerr << "zero fmas saturation: " << ((double)zero_fmas / (double)time) / 2.0
            << "\t min:" << ((double)zero_fmas / (double)max_time) / 2.0
            << "\t max:" << ((double)zero_fmas / (double)min_time) / 2.0
            << "\t ave:" << ((double)zero_fmas / average_cycles) / 2.0
            << std::endl;

        ASSERT_NO_FATAL_FAILURE(checkValues(alignedInput.get(), alignedOutput.get(), alignedWeights.get(),
            in_dims, out_dims, kern_info,
            make<Batch>(batch * loops), make<BatchBlock>(batch), make<OutFeatBlockSize>(OUTPUT_ACCEPTED_MOD)));
    }

    release(convInternal);
}

using namespace std::placeholders;
INSTANTIATE_TEST_CASE_P(PerfTestInst,
                        PerfTest,
                        ::testing::Values(
                            PerfTestSizes{InputDimensions{make<InputHeight>(13u), make<InputWidth>(13u), make<InputFeats>(16u*INPUT_ACCEPTED_MOD)},
                                          KernelDimensions{make<KernelHeight>(3u), make<KernelWidth>(3u), make<KernelFeats>(16u*INPUT_ACCEPTED_MOD)},
                                          make<OutputFeats>(OUTPUT_ACCEPTED_MOD*96u),
                                          105u},
                            PerfTestSizes{InputDimensions{make<InputHeight>(58u), make<InputWidth>(58u), make<InputFeats>(4u*INPUT_ACCEPTED_MOD)},
                                          KernelDimensions{make<KernelHeight>(5u), make<KernelWidth>(5u), make<KernelFeats>(4u*INPUT_ACCEPTED_MOD)},
                                          make<OutputFeats>(OUTPUT_ACCEPTED_MOD*67u),
                                          105u},
                            PerfTestSizes{InputDimensions{make<InputHeight>(48u), make<InputWidth>(48u), make<InputFeats>(16u*INPUT_ACCEPTED_MOD)},
                                          KernelDimensions{make<KernelHeight>(5u), make<KernelWidth>(5u), make<KernelFeats>(16u*INPUT_ACCEPTED_MOD)},
                                          make<OutputFeats>(OUTPUT_ACCEPTED_MOD*96u),
                                          105u},
                            PerfTestSizes{InputDimensions{make<InputHeight>(56u), make<InputWidth>(56u), make<InputFeats>(INPUT_ACCEPTED_MOD)},
                                          KernelDimensions{make<KernelHeight>(11u), make<KernelWidth>(11u), make<KernelFeats>(INPUT_ACCEPTED_MOD)},
                                          make<OutputFeats>(OUTPUT_ACCEPTED_MOD*24u),
                                          105u},
                            PerfTestSizes{InputDimensions{make<InputHeight>(48u), make<InputWidth>(48u), make<InputFeats>(16u*INPUT_ACCEPTED_MOD)},
                                          KernelDimensions{make<KernelHeight>(3u), make<KernelWidth>(3u), make<KernelFeats>(16u*INPUT_ACCEPTED_MOD)},
                                          make<OutputFeats>(OUTPUT_ACCEPTED_MOD*96u),
                                          105u}));

} //namespace

