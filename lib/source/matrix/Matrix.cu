#include "matrix_vector_op.hpp"
#include <curand_kernel.h>


namespace qlm
{
    // Default constructor
	Matrix::Matrix()
	{}

	// Parameterized constructor
	Matrix::Matrix(int rows, int columns, const int _stride) : rows(rows), columns(columns)
	{
		if (_stride > 0 && _stride >= columns)
			stride = _stride;
		else
			stride = columns;
		
		cudaMalloc(&data, stride * rows * sizeof(float));
	}

	// Copy constructor
	Matrix::Matrix(const Matrix& other) : columns(other.columns), rows(other.rows), stride(other.stride)
	{
		cudaMalloc(&data, stride * rows * sizeof(float));
        cudaMemcpy(data, other.data, stride * rows * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	// Destructor
	Matrix::~Matrix() 
	{
		rows = columns = stride = 0;
		if (data != nullptr)
			cudaFree(data);
	}

	// Setter for individual element (host to device)
	void Matrix::Set(int row, int col, float value) 
	{
		if (row >= 0 && row < rows && col >= 0 && col < columns)
		{
            data[row * stride + col] = value;
		}
	}

	// Getter for individual element
	float Matrix::Get(int row, int col) const 
	{
        float value = -1.0f; 
		if (row >= 0 && row < rows && col >= 0 && col < columns)
		{
			return data[row * stride + col];
		}
		return value;
	}

	// Getter for columns
	int Matrix::Columns() const
	{
		return columns;
	}

	// Getter for rows
	int Matrix::Rows() const
	{
		return rows;
	}

    int Matrix::Stride() const
	{
		return stride;
	}

    void Matrix::Alloc(const int rows, const int columns, const int stride)
    {
        if (data != nullptr)
            cudaFree(data);
        this->rows = rows;
        this->columns = columns;
		this->stride = (stride > 0 && stride >= columns) ? stride : columns;
        cudaMalloc(&data, rows * stride * sizeof(float));
    }
    
    void Matrix::FromCPU(const float* src, const int num_rows, const int num_stride)
    {
		if (rows != num_rows || stride != num_stride || data == nullptr)
        	Alloc(num_rows, num_stride, num_stride);

		cudaMemcpy(data, src, rows * num_stride * sizeof(float), cudaMemcpyHostToDevice);
    }
    
	void Matrix::ToCPU(float* dst, const int rows, const int num_stride) const
    {
        if (dst != nullptr && data != nullptr)
        {
			cudaMemcpy(dst, data, rows * num_stride * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
}