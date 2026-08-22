#pragma once

#include "matrix_vector_op.hpp"

namespace qlm
{
    // ---- Elementwise operation functors ----
    __device__ __forceinline__ float OpAdd(float a, float b) { return a + b; };
    __device__ __forceinline__ float OpSub(float a, float b) { return a - b; };
    __device__ __forceinline__ float OpMul(float a, float b) { return a * b; };
    __device__ __forceinline__ float OpDiv(float a, float b) { return a / b; };

    // ---- Generic elementwise kernel ----
    template<auto Op>
    __global__ void VectorElementwise_Cuda(const float* in0, const float* in1, float* out, const int length)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < length)
        {
            out[tid] = Op(in0[tid], in1[tid]);
        }
    }

    // ---- Generic host-side dispatcher ----
    template<MemType mem_type, auto Op>
    static inline void VectorProcessor_Elementwise(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst)
    {
        const int length = std::min(src0.Length(), src1.Length());

        if constexpr (mem_type == MemType::MEM_UM)
        {
            src0.PrefetchToGPU();
            src1.PrefetchToGPU();
            dst.PrefetchToGPU();
        }

        const int block_size = 256;
        const int num_blocks = (length + block_size - 1) / block_size;
        VectorElementwise_Cuda<Op><<<num_blocks, block_size>>>(src0.Data(), src1.Data(), dst.Data(), length);
        cudaDeviceSynchronize();
    }
}