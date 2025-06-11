#include "matrix.hpp"

#define TILE_SIZE 16
namespace qlm
{
    __global__ void MatrixDot_Cuda(const float* src0, const float* src1, float* dst, const int d0, const int d1, const int d2)
    {
        const int thread_row = threadIdx.y;
        const int thread_col = threadIdx.x;
        const int row = blockIdx.y * blockDim.y + thread_row;
        const int col = blockIdx.x * blockDim.x + thread_col;

        __shared__ float shared_src0[TILE_SIZE][TILE_SIZE];
        __shared__ float shared_src1[TILE_SIZE][TILE_SIZE];

        float sum = 0.0f;

        // Loop over tiles
        int num_tiles = (d1 + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < num_tiles; ++t)
        {
            shared_src0[thread_row][thread_col] = 0.0f;
            shared_src1[thread_row][thread_col] = 0.0f;

            int tiled_col = t * TILE_SIZE + thread_col;
            int tiled_row = t * TILE_SIZE + thread_row;

            // Load tile from src0
            if (row < d0 && tiled_col < d1)
                shared_src0[thread_row][thread_col] = src0[row * d1 + tiled_col];

            // Load tile from src1
            if (tiled_row < d1 && col < d2)
                shared_src1[thread_row][thread_col] = src1[tiled_row * d2 + col];

            __syncthreads();

            // Compute partial sum for this tile
            for (int k = 0; k < TILE_SIZE; ++k)
                sum += shared_src0[thread_row][k] * shared_src1[k][thread_col];

            __syncthreads();
        }

        // Write result
        if (row < d0 && col < d2)
            dst[row * d2 + col] = sum;
    }

    void qlm::Matrix::Dot(const Matrix &src, Matrix &dst) const
    {
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 num_blocks((src.columns + block_size.x - 1) / block_size.x, (rows + block_size.y - 1) / block_size.y);

        // Launch kernel
        MatrixDot_Cuda<<<num_blocks, block_size>>>(data, src.data, dst.data, rows, columns, src.columns);
        cudaDeviceSynchronize();
    }
} // namespace qlm