#include "matrix.hpp"

namespace qlm
{
    struct GPUMatrix
    {
        float *data;
        int columns;
        int rows;
        size_t stride;

        GPUMatrix(float *data, int rows, int columns, size_t stride)
            : data(data), columns(columns), rows(rows), stride(stride) {}

        __device__ float Get(int r, int c) const
        {
            return data[r * stride + c];
        }
        __device__ void Set(int r, int c, float value)
        {
            data[r * stride + c] = value;
        }
    };

    __global__ void MatrixDot_Cuda(const GPUMatrix &src0, const GPUMatrix &src1, GPUMatrix &dst)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;

        for (int i = 0; i < src0.columns; i++)
        {
           sum += src0.Get(row, i) * src1.Get(i, col);
        }

        dst.Set(row, col, sum);
    }

    void qlm::Matrix::Dot(const Matrix &src, Matrix &dst) const
    {
        dim3 block_size(16, 16);
        dim3 num_blocks((columns + block_size.x - 1) / block_size.x, (rows + block_size.y - 1) / block_size.y);

        GPUMatrix src0_gpu(data, this->Rows(), this->Columns(), this->Stride());
        GPUMatrix src1_gpu(src.data, src.Rows(), src.Columns(), src.Stride());
        GPUMatrix dst_gpu(dst.data, dst.Rows(), dst.Columns(), dst.Stride());
        // Launch kernel
        MatrixDot_Cuda<<<num_blocks, block_size>>>(src0_gpu, src1_gpu, dst_gpu);
        cudaDeviceSynchronize();
    }
} // namespace qlm