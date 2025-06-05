#include "ref_matrix.hpp"
#include <random>
#include <iostream>
#include <iomanip>
namespace test
{
    Matrix::Matrix() : data(nullptr), columns(0), rows(0)
	{}

	// Parameterized constructor
	Matrix::Matrix(int r, int c) : columns(c), rows(r) 
	{
		data = new float[columns * rows];
	}

	// Copy constructor
	Matrix::Matrix(const Matrix& other) : columns(other.columns), rows(other.rows) 
	{
		data = new float[columns * rows];
		for (int i = 0; i < columns * rows; ++i) {
			data[i] = other.data[i];
		}
	}

	// Destructor
	Matrix::~Matrix() 
	{
		rows = columns = 0;
		if (data != nullptr)
			delete[] data;
	}

	// Setter for individual element
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
		if (row >= 0 && row < rows && col >= 0 && col < columns)
		{
			return data[row * columns + col];
		}
		return std::numeric_limits<float>::signaling_NaN();
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

    void Matrix::RandomInit(const float min_value, const float max_value)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(min_value, max_value);

		for (int r = 0; r < rows; r++)
		{
            for (int c = 0; c < columns; c++)
            {
                int i = r * columns + c;
                this->Set(r, c, dis(gen));
            }
		}
	}

	void Matrix::Print() const 
	{
		for (int r = 0; r < rows; r++)
		{
			int number_digits = 5;

			for (int c = 0; c < columns; c++)
			{
				float element = this->Get(r, c);

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

}