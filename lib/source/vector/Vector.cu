#include "vector.hpp"
#include <random>
#include <iostream>
#include <iomanip>
#include <functional>
#include <curand_kernel.h>


namespace qlm
{
	// Default constructor
	Vector::Vector() : data(nullptr), length(0)
	{}

	// Parameterized constructor
	Vector::Vector(int length) : length(length)
	{
		cudaMalloc(&data, length * sizeof(float));
	}

	// Copy constructor
	Vector::Vector(const Vector& other) : length(other.length)
	{
		cudaMalloc(&data, length * sizeof(float));
		cudaMemcpy(data, other.data, length * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	// Destructor
	Vector::~Vector()
	{
		if (data != nullptr)
			cudaFree(data);
	}

	// Setter for individual element (host to device)
    void Vector::Set(const int i, const float value) 
	{
        if (i >= 0 && i < length)
        {
            cudaMemcpy(data + i, &value, sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    // Getter for individual element (device to host)
    float Vector::Get(const int i) const
    {
        float value = std::numeric_limits<float>::signaling_NaN();
        if (i >= 0 && i < length)
        {
            cudaMemcpy(&value, data + i, sizeof(float), cudaMemcpyDeviceToHost);
        }
        return value;
    }
	
	// Getter for length
	int Vector::Length() const
	{
		return length;
	}
	// allocate memory
	void Vector::Alloc(const int len)
	{
		if (data != nullptr)
		{
			cudaFree(data);
        }

        cudaMalloc(&data, len * sizeof(float));
		length = len;
	}

	// Random initialization of vector elements
	__global__ void RandomInitKernel(float* data, const int length, const float min_value, const float max_value, const unsigned long long seed) 
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < length) {
			curandState state;
			curand_init(seed, idx, 0, &state); // Initialize CURAND
			data[idx] = curand_uniform(&state) * (max_value - min_value) + min_value;
		}
	}
	void Vector::RandomInit(const float min_value, const float max_value)
	{
		// Determine grid and block sizes
		const int blockSize = 256;
		const int gridSize = (length + blockSize - 1) / blockSize;

		// Generate a random seed on host
		std::random_device rd;
		unsigned long long seed = rd();

		// Launch the kernel
		RandomInitKernel<<<gridSize, blockSize>>>(data, length, min_value, max_value, seed);

		// Synchronize to make sure initialization is complete
    	cudaDeviceSynchronize();
	}

	void Vector::Print() const
	{
		// Allocate host buffer
		std::vector<float> host_data(length);
		
		// Copy device data to host
		cudaMemcpy(host_data.data(), data, length * sizeof(float), cudaMemcpyDeviceToHost);

		// Print with formatting (using original CPU logic)
		int number_digits = 5;
		for (int l = 0; l < length; l++) 
		{
			const float element = host_data[l];
			
			if (element != 0) 
			{
				int digits = static_cast<int>(std::log10(std::abs(element))) + 1;
				number_digits = digits >= 5 ? 0 : 5 - digits;
			}
			
			std::cout << std::fixed << std::setprecision(number_digits) << element << " ";
		}
		std::cout << std::endl;
	}
}