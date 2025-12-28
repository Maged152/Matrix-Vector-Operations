#include "matrix_vector_op.hpp"

#define BLOCK_SIZE 256

namespace qlm
{
    __device__ void WarpReduce(volatile float* partial_sum, const int tid) 
    {
        partial_sum[tid] += partial_sum[tid + 32];
        partial_sum[tid] += partial_sum[tid + 16];
        partial_sum[tid] += partial_sum[tid + 8];
        partial_sum[tid] += partial_sum[tid + 4];
        partial_sum[tid] += partial_sum[tid + 2];
        partial_sum[tid] += partial_sum[tid + 1];
    }

    __global__ void VectorSum_Cuda(const float* in, const int length, float* result)
    {
        const int tid = threadIdx.x;
        const int gid = blockIdx.x * blockDim.x * 2 + tid;

        __shared__ float partial_sum[BLOCK_SIZE];

        // Load elements & do first add of reduction
        const float second_element = (gid + blockDim.x < length) ? in[gid + blockDim.x] : 0.0f;
        partial_sum[tid] = in[gid] + second_element;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 32; s >>= 1) 
        { 
            if (tid < s)
            {
                partial_sum[tid] += partial_sum[tid + s];
            }
            __syncthreads();
        }

        // last warp
        if (tid < 32) 
            WarpReduce(partial_sum, tid);

        if (tid == 0) 
        {
            atomicAdd(result, partial_sum[0]);
        }
    }

    void qlm::Sum(const Vector &src, DeviceFloat& result)
	{
        const int length = src.Length();
        // Launch kernel
        const int block_size = BLOCK_SIZE;
        const int num_blocks = (length + (block_size * 2) - 1) / (block_size * 2);
        
        VectorSum_Cuda<<<num_blocks, block_size>>>(src.data, length, result.mem.data);
        cudaDeviceSynchronize();
	}
}