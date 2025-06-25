#include "matrix_vector_op.hpp"
#include <curand_kernel.h>


namespace qlm
{
     __global__ void Div_Cuda(float* in, const int length)
    {
        in[0] /= static_cast<float>(length);
    }
    
    void qlm::Mean(const Vector &src, DeviceFloat& result)
	{
        qlm::Sum(src, result);
        const int length = src.Length();  
        Div_Cuda<<<1, 1>>>(result.mem.data, length);
        cudaDeviceSynchronize();
    }
}