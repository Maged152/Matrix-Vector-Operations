#include "matrix_vector_op.hpp"
#include <stdexcept>

extern __constant__ unsigned char CudaConstMem_ptr[];
namespace qlm
{
    // Full convolution: output size = src + ker - 1
    __global__ void MatrixConvFull_Cuda(const float* src, const int src_rows, const int src_cols, const int src_stride,
                                         const int ker_rows, const int ker_cols,
                                         float* dst, const int dst_rows, const int dst_cols, const int dst_stride)
    {
        const int r = blockIdx.y * blockDim.y + threadIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= dst_rows || c >= dst_cols) return;

        float sum = 0.0f;
        for (int kr = 0; kr < ker_rows; ++kr)
        {
            const int src_r = r + kr - (ker_rows - 1);
            for (int kc = 0; kc < ker_cols; ++kc)
            {
                const int src_c = c + kc - (ker_cols - 1);

                if (src_r >= 0 && src_r < src_rows && src_c >= 0 && src_c < src_cols)
                {
                    const float pixel = src[src_r * src_stride + src_c];
                    // follow reference: use kernel as-is (no flip)
                    const float kval = reinterpret_cast<const float*>(CudaConstMem_ptr)[kr * ker_cols + kc];
                    sum += pixel * kval;
                }
            }
        }

        dst[r * dst_stride + c] = sum;
    }

    // Same convolution: output size = src, center kernel
    __global__ void MatrixConvSame_Cuda(const float* src, const int src_rows, const int src_cols, const int src_stride,
                                        const int ker_rows, const int ker_cols,
                                        float* dst, const int dst_rows, const int dst_cols, const int dst_stride)
    {
        const int r = blockIdx.y * blockDim.y + threadIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= dst_rows || c >= dst_cols) return;

        const int center_y = ker_rows / 2;
        const int center_x = ker_cols / 2;

        float sum = 0.0f;
        for (int kr = 0; kr < ker_rows; ++kr)
        {
            const int src_r = r + (kr - center_y);
            for (int kc = 0; kc < ker_cols; ++kc)
            {
                const int src_c = c + (kc - center_x);

                if (src_r >= 0 && src_r < src_rows && src_c >= 0 && src_c < src_cols)
                {
                    const float pixel = src[src_r * src_stride + src_c];
                    const float kval = reinterpret_cast<const float*>(CudaConstMem_ptr)[kr * ker_cols + kc];
                    sum += pixel * kval;
                }
            }
        }

        dst[r * dst_stride + c] = sum;
    }

    // Valid convolution: kernel fully inside src
    __global__ void MatrixConvValid_Cuda(const float* src, const int src_rows, const int src_stride,
                                         const int ker_rows, const int ker_cols,
                                         float* dst, const int dst_rows, const int dst_cols, const int dst_stride)
    {
        const int r = blockIdx.y * blockDim.y + threadIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= dst_rows || c >= dst_cols) return;

        float sum = 0.0f;
        for (int kr = 0; kr < ker_rows; ++kr)
        {
            const int src_r = r + kr;
            for (int kc = 0; kc < ker_cols; ++kc)
            {
                const int src_c = c + kc;
                const float pixel = src[src_r * src_stride + src_c];
                const float kval = reinterpret_cast<const float*>(CudaConstMem_ptr)[kr * ker_cols + kc];
                sum += pixel * kval;
            }
        }

        dst[r * dst_stride + c] = sum;
    }

    void qlm::Conv(const Matrix& src, const Matrix& kernel, Matrix& dst, const qlm::ConvMode mode)
    {
        const int src_rows = src.rows;
        const int src_cols = src.columns;
        const int src_stride = src.stride;

        const int ker_rows = kernel.rows;
        const int ker_cols = kernel.columns;

        const int dst_rows = dst.rows;
        const int dst_cols = dst.columns;
        const int dst_stride = dst.stride;

        const size_t kernel_size = static_cast<size_t>(ker_rows) * static_cast<size_t>(ker_cols) * sizeof(float);
        if (kernel_size > USED_CONST_MEM_BYTES)
            throw std::runtime_error("Kernel size exceeds constant memory limit.");

        // copy kernel to constant memory
        cudaMemcpyToSymbol(CudaConstMem_ptr, kernel.data, kernel_size);

        // launch params
        const int num_threads_per_block = 16;
        const int num_blocks_x = (dst_cols + num_threads_per_block - 1) / num_threads_per_block;
        const int num_blocks_y = (dst_rows + num_threads_per_block - 1) / num_threads_per_block;

        const dim3 block(num_threads_per_block, num_threads_per_block);
        const dim3 grid(num_blocks_x, num_blocks_y);

        if (mode == ConvMode::FULL)
        {
            MatrixConvFull_Cuda<<<grid, block>>>(src.data, src_rows, src_cols, src_stride, ker_rows, ker_cols, dst.data, dst_rows, dst_cols, dst_stride);
        }
        else if (mode == ConvMode::SAME)
        {
            MatrixConvSame_Cuda<<<grid, block>>>(src.data, src_rows, src_cols, src_stride, ker_rows, ker_cols, dst.data, dst_rows, dst_cols, dst_stride);
        }
        else // VALID
        {
            MatrixConvValid_Cuda<<<grid, block>>>(src.data, src_rows, src_stride, ker_rows, ker_cols, dst.data, dst_rows, dst_cols, dst_stride);
        }

        cudaDeviceSynchronize();
    }
}
