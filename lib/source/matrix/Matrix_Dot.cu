#include "matrix.hpp"

namespace qlm
{
    __global__ void MatrixDot_Cuda(const float* src0, const float* src1, float* dst, const int d0, const int d1, const int d2)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;


        if (row >= d0 || col >= d2)
            return; // Out of bounds check

        for (int i = 0; i < d1; i++)
        {

            sum += src0[row * d1 + i] * src1[i * d2 + col];
        }

        dst[row * d2 + col] = sum;
    }

    void qlm::Matrix::Dot(const Matrix &src, Matrix &dst) const
    {
        dim3 block_size(16, 16);
        dim3 num_blocks((src.columns + block_size.x - 1) / block_size.x, (rows + block_size.y - 1) / block_size.y);

        // Launch kernel
        MatrixDot_Cuda<<<num_blocks, block_size>>>(data, src.data, dst.data, rows, columns, src.columns);
        cudaDeviceSynchronize();
    }
} // namespace qlm