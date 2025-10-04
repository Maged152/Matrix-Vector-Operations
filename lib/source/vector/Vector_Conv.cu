#include "matrix_vector_op.hpp"

namespace qlm
{
	__global__ void VectorConvFull_Cuda(const float* input, const int input_length, const float* kernel, const int kernel_length, float* output, const int output_length)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{
			const int input_idx = idx + k - (kernel_length - 1);

			if (input_idx >= 0 && input_idx < input_length)
			{
				sum += input[input_idx] * kernel[kernel_length - 1 - k];
			}
		}

		output[idx] = sum;
	}

	__global__ void VectorConvSame_Cuda(const float* input, const int input_length, const float* kernel, const int kernel_length, float* output, const int output_length)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{
			const int input_idx = idx + k - (kernel_length / 2);

			if (input_idx >= 0 && input_idx < input_length)
			{
				sum += input[input_idx] * kernel[kernel_length - 1 - k];
			}
		}

		output[idx] = sum;
	}

	__global__ void VectorConvValid_Cuda(const float* input, const int input_length, const float* kernel, const int kernel_length, float* output, const int output_length)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{
			const int input_idx = idx + k;

			sum += input[input_idx] * kernel[kernel_length - 1 - k];
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

		if (mode == ConvMode::FULL)
			VectorConvFull_Cuda<<<num_blocks, block_size>>>(input.data, input_length, kernel.data, kernel_length, output.data, output_length);
		else if (mode == ConvMode::SAME)
			VectorConvSame_Cuda<<<num_blocks, block_size>>>(input.data, input_length, kernel.data, kernel_length, output.data, output_length);
		else // mode == ConvMode::VALID
			VectorConvValid_Cuda<<<num_blocks, block_size>>>(input.data, input_length, kernel.data, kernel_length, output.data, output_length);

		cudaDeviceSynchronize(); // Ensure the kernel execution is complete
  
	}
}