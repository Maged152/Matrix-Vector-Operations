#include "matrix_vector_op.hpp"
#include <stdexcept>

extern __constant__ unsigned char CudaConstMem_ptr[];
namespace qlm
{
	__global__ void VectorConvFull_Cuda(const float* input, const int input_length, const int kernel_length, float* output, const int output_length)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{
			const int input_idx = idx + k - (kernel_length - 1);

			if (input_idx >= 0 && input_idx < input_length)
			{
				sum += input[input_idx] * reinterpret_cast<const float*>(CudaConstMem_ptr)[kernel_length - 1 - k];
			}
		}

		output[idx] = sum;
	}

	__global__ void VectorConvSame_CudaSM(const float* input, const int input_length, const int kernel_length, float* output, const int output_length)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		extern __shared__ float shared_mem[];

		const int r = kernel_length >> 1;
		const int d = r << 1;

		// center element
		shared_mem[threadIdx.x + r] = input[idx];
		
		// lower halo
		if (threadIdx.x < r)
		{
			shared_mem[threadIdx.x] = idx < r ? 0.0f : input[idx - r];
		}
		
		// upper halo
		if (threadIdx.x + r >= blockDim.x)
		{
			shared_mem[threadIdx.x + d] = (idx + r) >= input_length ? 0.0f : input[idx + r];
		}

		__syncthreads();

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{	
			sum += shared_mem[threadIdx.x + k] * reinterpret_cast<const float*>(CudaConstMem_ptr)[kernel_length - 1 - k];
		}

		output[idx] = sum;
	}

	__global__ void VectorConvSame_Cuda(const float* input, const int input_length, const int kernel_length, float* output, const int output_length)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{
			const int input_idx = idx + k - (kernel_length / 2);

			if (input_idx >= 0 && input_idx < input_length)
			{
				sum += input[input_idx] * reinterpret_cast<const float*>(CudaConstMem_ptr)[kernel_length - 1 - k];
			}
		}

		output[idx] = sum;
	}

	__global__ void VectorConvValid_Cuda(const float* input, const int input_length, const int kernel_length, float* output, const int output_length)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= output_length) return;

		float sum = 0.0f;

		for (int k = 0; k < kernel_length; k++)
		{
			const int input_idx = idx + k;

			sum += input[input_idx] * reinterpret_cast<const float*>(CudaConstMem_ptr)[kernel_length - 1 - k];
		}

		output[idx] = sum;
	}

    void qlm::Conv(const Vector& input, const Vector& kernel, Vector& output, const qlm::ConvMode mode)
	{
		const int input_length = input.Length();
		const int kernel_length = kernel.Length();
		const int output_length = output.Length();

		// copy kernel to constant memory
		if (kernel_length * sizeof(float) > USED_CONST_MEM_BYTES)
			throw std::runtime_error("Kernel size exceeds constant memory limit.");

		// limitation: kernel_length must be larger than input_length
		if (kernel_length > input_length)
			throw std::runtime_error("kernel length must be less than or equal to input length.");

		cudaMemcpyToSymbol(CudaConstMem_ptr, kernel.data, kernel_length * sizeof(float));
		
		const int block_size = 256;
		const int num_blocks = (output_length + block_size - 1) / block_size;
		const int extra_size = kernel_length - 1;
		const int shared_mem_size = (block_size + extra_size) * sizeof(float);

		bool use_shared_mem = output_length % block_size == 0 ; // only use shared memory when all threads in the last block are used

		// Launch kernel
		if (mode == ConvMode::FULL)
			VectorConvFull_Cuda<<<num_blocks, block_size, shared_mem_size>>>(input.data, input_length, kernel_length, output.data, output_length);
		else if (mode == ConvMode::SAME) {
			if (use_shared_mem)
				VectorConvSame_CudaSM<<<num_blocks, block_size, shared_mem_size>>>(input.data, input_length, kernel_length, output.data, output_length);
			else
				VectorConvSame_Cuda<<<num_blocks, block_size>>>(input.data, input_length, kernel_length, output.data, output_length);
		}
		else // mode == ConvMode::VALID
			VectorConvValid_Cuda<<<num_blocks, block_size>>>(input.data, input_length, kernel_length, output.data, output_length);

		cudaDeviceSynchronize(); // Ensure the kernel execution is complete
  
	}
}