#include "matrix_vector_op.hpp"

namespace qlm
{
	__global__ void VectorConv_Cuda(const float* input, const int input_length, const float* kernel, const int kernel_length, float* output, const int output_length, const int mode)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{
			int input_idx;

			if (mode == static_cast<int>(qlm::ConvMode::FULL))
			{
				// Full mode: No restrictions on input_idx
				input_idx = idx + k - (kernel_length - 1);
			}
			else if (mode == static_cast<int>(qlm::ConvMode::SAME))
			{
				// Same mode: Center the kernel over the input
				input_idx = idx + k - (kernel_length / 2);
			}
			else if (mode == static_cast<int>(qlm::ConvMode::VALID))
			{
				// Valid mode: Only compute where the kernel fully overlaps with the input
				input_idx = idx + k;
			}

			if (input_idx >= 0 && input_idx < input_length)
			{
				sum += input[input_idx] * kernel[kernel_length - 1 - k];
			}
		}

		output[idx] = sum;
	}

    void qlm::Conv(const Vector& input, const Vector& kernel, Vector& output, const qlm::ConvMode mode)
	{
		const int input_length = input.Length();
		const int kernel_length = kernel.Length();
		const int output_length = output.Length();

		// Launch kernel
		const int block_size = 256;
		const int num_blocks = (output_length + block_size - 1) / block_size;
		VectorConv_Cuda<<<num_blocks, block_size>>>(input.data, input_length, kernel.data, kernel_length, output.data, output_length, static_cast<int>(mode));
		cudaDeviceSynchronize(); // Ensure the kernel execution is complete
  
	}
}