#include "vector.hpp"
#include <algorithm>

#define BLOCK_SIZE 256

namespace qlm
{
    __global__ void VectorSum_Cuda(const float* in, const int length, float* result)
    {
        const int tid = threadIdx.x;
        const int gid = blockIdx.x * blockDim.x + tid;

        __shared__ float partial_sum[BLOCK_SIZE];
        partial_sum[tid] = (gid < length) ? in[gid] : 0.0f;

	    __syncthreads();

        for (int s = 1; s < blockDim.x; s *= 2) 
        {    
            const int index = 2 * s * tid;
            if (index < blockDim.x)
            {
                partial_sum[index] += partial_sum[index + s];
            }
            __syncthreads();
        }

        if (tid == 0) 
        {
            result[blockIdx.x] = partial_sum[0];
        }
    }

    __global__ void VectorSumBlock_Cuda(const float* in, const int length, float* result)
    {
        const int tid = threadIdx.x;

        __shared__ float partial_sum[BLOCK_SIZE];
        partial_sum[tid] = (tid < length) ? in[tid] : 0.0f;

	    __syncthreads();

        for (int s = 1; s < blockDim.x; s *= 2) 
        {    
            const int index = 2 * s * tid;
            if (index < blockDim.x)
            {
                partial_sum[index] += partial_sum[index + s];
            }
            __syncthreads();
        }

        if (tid == 0) 
        {
            *result += partial_sum[0];
        }
    }


    void qlm::Vector::Sum(DeviceMemory& result) const
	{
        // Launch kernel
        const int block_size = BLOCK_SIZE;
        const int num_blocks = (length + block_size - 1) / block_size;

        // allocate device memory for the result
        float* sum_result;
        cudaMalloc(&sum_result, num_blocks * sizeof(float));
        
        // First reduction: input -> partial sums
        VectorSum_Cuda<<<num_blocks, block_size>>>(data, length, sum_result);
        cudaDeviceSynchronize();

        // Second reduction: partial sums -> final sum
        for (int i = 0; i < num_blocks; i += block_size) 
        {
            const int cur_length = std::min(block_size, num_blocks - i);
            VectorSumBlock_Cuda<<<1, block_size>>>(&sum_result[i], cur_length, result.data);
            cudaDeviceSynchronize();
        }
        
        cudaFree(sum_result); // Free the device memory
	}
}