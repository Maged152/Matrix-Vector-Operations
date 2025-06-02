#include "vector.hpp"

namespace qlm
{
    __global__ void VectorAdd_Cuda(const float* in0, const float* in1, float* out, const int length)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < length)
        {
            out[tid] = in0[tid] + in1[tid];
        }
    }

    void qlm::Vector::Add(const Vector &src, Vector &dst) const
	{
        const float* in0 = this->data;
        const float* in1 = src.data;
        float* out = dst.data;

        float *in0_cuda, *in1_cuda, *out_cuda;
        const size_t bytes = length * sizeof(float);

        // Allocate device memory
        cudaMalloc(&in0_cuda, bytes);
        cudaMalloc(&in1_cuda, bytes);
        cudaMalloc(&out_cuda, bytes);

        // Copy data to device
        cudaMemcpy(in0_cuda, in0, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(in1_cuda, in1, bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        const int block_size = 256;
        const int num_blocks = (length + block_size - 1) / block_size;
        VectorAdd_Cuda<<<num_blocks, block_size>>>(in0_cuda, in1_cuda, out_cuda, length);

        // Copy result back to host
        cudaMemcpy(out, out_cuda, bytes, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(in0_cuda);
        cudaFree(in1_cuda);
        cudaFree(out_cuda);
	}
}