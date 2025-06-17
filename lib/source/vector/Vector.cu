#include "matrix_vector_op.hpp"
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

	// Setter for individual element
    void Vector::Set(const int i, const float value) 
	{
        if (i >= 0 && i < length)
        {
            data[i] = value;
        }
    }

    // Getter for individual element
    float Vector::Get(const int i) const
    {
        float value = std::numeric_limits<float>::signaling_NaN();
        if (i >= 0 && i < length)
        {
			value = data[i];
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

	// copy data from CPU to GPU
	void Vector::FromCPU(const float* src, const int len)
	{
		if (data != nullptr)
		{
			cudaFree(data);
		}
		
		length = len;
		cudaMalloc(&data, length * sizeof(float));
		cudaMemcpy(data, src, length * sizeof(float), cudaMemcpyHostToDevice);
	}

	// copy data from GPU to CPU
	void Vector::ToCPU(float* dst, const int len) const
	{
		if (data != nullptr && dst != nullptr && len == length)
		{
			cudaMemcpy(dst, data, length * sizeof(float), cudaMemcpyDeviceToHost);
		}
	}
}