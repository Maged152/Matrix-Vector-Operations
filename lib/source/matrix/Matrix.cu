#include "matrix.hpp"
#include <curand_kernel.h>


namespace qlm
{
    // Default constructor
	Matrix::Matrix()
	{}

	// Parameterized constructor
	Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns) 
	{
		cudaMallocPitch(&data, &stride, columns * sizeof(float), rows);
		stride /= sizeof(float);
	}

	// Copy constructor
	Matrix::Matrix(const Matrix& other) : columns(other.columns), rows(other.rows) 
	{
		cudaMallocPitch(&data, &stride, columns * sizeof(float), rows);
        cudaMemcpy2D(data, stride, other.data, other.stride, columns * sizeof(float), rows, cudaMemcpyDeviceToDevice);
		stride /= sizeof(float);
	}

	// Destructor
	Matrix::~Matrix() 
	{
		rows = columns = stride =0;
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

    void Matrix::Alloc(const int rows, const int columns)
    {
        if (data != nullptr)
            cudaFree(data);
        this->rows = rows;
        this->columns = columns;
        cudaMallocPitch(&data, &stride, columns * sizeof(float), rows);
		stride /= sizeof(float);
    }
    
    void Matrix::FromCPU(const float* src, const int rows, const int columns)
    {
        Alloc(rows, columns);
        if (src != nullptr)
        {
            cudaMemcpy2D(data, stride * sizeof(float), src, columns * sizeof(float), columns * sizeof(float), rows, cudaMemcpyHostToDevice);
        }
    }
    
	void  Matrix::ToCPU(float* dst, const int rows, const int columns) const
    {
        if (dst != nullptr && data != nullptr)
        {
            cudaMemcpy2D(dst, columns * sizeof(float), data, stride * sizeof(float), columns * sizeof(float), rows, cudaMemcpyDeviceToHost);
        }
    }
}