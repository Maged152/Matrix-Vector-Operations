#include "vector.hpp"

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

    void qlm::Vector::Sum(DeviceMemory& result) const
	{
        // Launch kernel
        const int block_size = BLOCK_SIZE;
        const int num_blocks = (length + block_size - 1) / block_size;
        // allocate device memory for the result
        float* sum_result;
        cudaMalloc(&sum_result, num_blocks * sizeof(float));
        
        VectorSum_Cuda<<<num_blocks, block_size>>>(data, length, sum_result);
        // VectorSum_Cuda<<<1, num_blocks>>>(sum_result, num_blocks, result.data);

        cudaDeviceSynchronize(); // Ensure the kernel execution is complete

        float res = 0;
        for (int i = 0; i < num_blocks; ++i) 
        {
            float block_sum;
            cudaMemcpy(&block_sum, &sum_result[i], sizeof(float), cudaMemcpyDeviceToHost);
            res += block_sum;
        }
        // Copy the result back to the host
        result.FromCPU(&res); // Copy the final sum to the result device memory

        cudaFree(sum_result); // Free the device memory
	}
}