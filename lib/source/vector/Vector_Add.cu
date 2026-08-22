#include "matrix_vector_op.hpp"

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

    template<MemType mem_type>
    void Add(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst)
	{
        const int length = std::min(src0.Length(), src1.Length());

        if constexpr (mem_type == qlm::MemType::MEM_UM)
        {
            // Prefetch data to GPU
            src0.PrefetchToGPU();
            src1.PrefetchToGPU();
            dst.PrefetchToGPU();
        }
        
        // Launch kernel
        const int block_size = 256;
        const int num_blocks = (length + block_size - 1) / block_size;
        VectorAdd_Cuda<<<num_blocks, block_size>>>(src0.Data(), src1.Data(), dst.Data(), length);
        cudaDeviceSynchronize(); // Ensure the kernel execution is complete
	}

    template void Add<MemType::MEM_GPU>(const Vector<MemType::MEM_GPU>&, const Vector<MemType::MEM_GPU>&, Vector<MemType::MEM_GPU>&);
    template void Add<MemType::MEM_UM>(const Vector<MemType::MEM_UM>&, const Vector<MemType::MEM_UM>&, Vector<MemType::MEM_UM>&);
}