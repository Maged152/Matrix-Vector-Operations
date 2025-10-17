#include "matrix_vector_op.hpp"
#include <curand_kernel.h>


namespace qlm
{
    // Default constructor
	Matrix::Matrix()
	{}

	// Parameterized constructor
	Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns) 
	{
		cudaMalloc(&data, rows * columns * sizeof(float));
	}

	// Copy constructor
	Matrix::Matrix(const Matrix& other) : columns(other.columns), rows(other.rows) 
	{
		cudaMalloc(&data, rows * columns * sizeof(float));
        cudaMemcpy(data, other.data, rows * columns * sizeof(float), cudaMemcpyDeviceToDevice);
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
            data[row * columns + col] = value;
		}
	}

	// Getter for individual element
	float Matrix::Get(int row, int col) const 
	{
        float value = -1.0f; 
		if (row >= 0 && row < rows && col >= 0 && col < columns)
		{
			return data[row * columns + col];
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

    void Matrix::Alloc(const int rows, const int columns)
    {
        if (data != nullptr)
            cudaFree(data);
        this->rows = rows;
        this->columns = columns;
        cudaMalloc(&data, rows * columns * sizeof(float));
    }
    
    void Matrix::FromCPU(const float* src, const int num_rows, const int num_columns)
    {
		if (rows != num_rows || columns != num_columns || data == nullptr)
        	Alloc(num_rows, num_columns);

		cudaMemcpy(data, src, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
    }
    
	void Matrix::ToCPU(float* dst, const int rows, const int columns) const
    {
        if (dst != nullptr && data != nullptr)
        {
			cudaMemcpy(dst, data, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
}