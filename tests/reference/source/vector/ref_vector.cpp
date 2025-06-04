#include "ref_vector.hpp"
#include <random>
#include <iostream>
#include <iomanip>


namespace test
{
	// Default constructor
	Vector::Vector() : data(nullptr), length(0)
	{}

	// Parameterized constructor
	Vector::Vector(int length) : length(length)
	{
		data = new float[length];
	}

	// Copy constructor
	Vector::Vector(const Vector& other) : length(other.length)
	{
		data = new float[length];
		for (int i = 0; i < length; ++i) {
			data[i] = other.data[i];
		}
	}

	// Destructor
	Vector::~Vector()
	{
		if (data != nullptr)
			delete[] data;
	}

	// Setter for individual element
	void Vector::Set(const int i, const float value) {
		if (i >= 0 && i < length)
		{
			data[i] = value;
		}
	}

	// Getter for individual element
	float Vector::Get(const int i) const
	{
		if (i >= 0 && i <length)
		{
			return data[i];
		}
		return std::numeric_limits<float>::signaling_NaN();
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
			delete[] data;
		}

		data = new float[len];
		length = len;
	}

	// Vector helper functions
	void Vector::RandomInit(const float min_value, const float max_value)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(min_value, max_value);

		for (int i = 0; i < length; i++)
		{
			this->Set(i, dis(gen));
		}
	}

	void Vector::Print() const
	{
		int number_digits = 5;

		for (int l = 0; l < length; l++)
		{
			float element = this->Get(l);

			if (element != 0)
			{
				int digits = static_cast<int>(std::log10(std::abs(element))) + 1;
				number_digits = digits >= 5 ? 0 : 5 - digits;
			}

			std::cout << std::fixed << std::setprecision(number_digits) << element << " ";
		}
	}
}