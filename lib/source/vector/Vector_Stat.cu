#include "matrix_vector_op.hpp"
#include <curand_kernel.h>


namespace qlm
{
     __global__ void Div_Cuda(float* in, const int length)
    {
        in[0] /= static_cast<float>(length);
    }
    
    template<MemType mem_type>
    void Mean(const Vector<mem_type>& src, Array<1, mem_type>& result)
	{
        Sum(src, result);
        const int length = src.Length();  
        Div_Cuda<<<1, 1>>>(result.Data(), length);
        cudaDeviceSynchronize();
    }

    template void Mean<MemType::MEM_GPU>(const Vector<MemType::MEM_GPU>&, Array<1, MemType::MEM_GPU>&);
    template void Mean<MemType::MEM_UM>(const Vector<MemType::MEM_UM>&, Array<1, MemType::MEM_UM>&);
}