#include "thread_pool.hpp"
#include "reference.hpp"
#include <string>
#include "print_types.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>

#define ANSI_TXT_GRN "\033[0;32m"
#define ANSI_TXT_RED "\033[0;31m"
#define ANSI_TXT_MGT "\033[0;35m" //Magenta
#define ANSI_TXT_DFT "\033[0;0m" //Console default

#define GTEST_BOX      "[Parameters] "
#define GTEST_BOX_TIME "[Time      ] "
#define GTEST_BOX_FAST "[Faster    ] "
#define GTEST_BOX_SLOW "[Slower    ] "

#define COUT_GTEST ANSI_TXT_GRN << GTEST_BOX 
#define COUT_GTEST_TIME ANSI_TXT_GRN << GTEST_BOX_TIME
#define COUT_GTEST_FAST ANSI_TXT_GRN << GTEST_BOX_FAST 
#define COUT_GTEST_SLOW ANSI_TXT_GRN << GTEST_BOX_SLOW 

#define COUT_GTEST_MGT COUT_GTEST << ANSI_TXT_MGT
#define COUT_GTEST_MGT_TIME COUT_GTEST_TIME << ANSI_TXT_MGT
#define COUT_GTEST_GRN_FAST COUT_GTEST_FAST << ANSI_TXT_GRN
#define COUT_GTEST_RED_SLOW COUT_GTEST_SLOW << ANSI_TXT_RED

namespace test
{
	// float to int
	inline void Float2Int(qlm::Vector& in)
	{
		for (int i = 0; i < in.Length(); i++)
		{
			float element = static_cast<float>(static_cast<int>(in.Get(i)));
			in.Set(i, element);
		}
	}
	// print
	template<typename T>
	inline void PrintParameter(T parameter, const std::string& para_name)
	{
		std::cout << COUT_GTEST_MGT << para_name 
			                        << " = " 
			                        << parameter 
			                        << ANSI_TXT_DFT << std::endl;
	}

	inline void PrintTime(const qlm::Timer<qlm::usec>& cpu, const qlm::Timer<qlm::usec>& gpu)
	{
		std::cout << COUT_GTEST_MGT_TIME << "gpu time"
			                             << " = "
			                             << gpu.ElapsedString()
		                                 << ANSI_TXT_DFT << std::endl;

		std::cout << COUT_GTEST_MGT_TIME << "cpu time"
			                             << " = "
			                             << cpu.ElapsedString()
			                             << ANSI_TXT_DFT << std::endl;

		if (gpu.Elapsed() < cpu.Elapsed())
		{
			std::cout << COUT_GTEST_GRN_FAST << "faster by "
				                             << " = "
				                             << ((cpu.Elapsed() - gpu.Elapsed()) / gpu.Elapsed()) * 100
				                             << " %"
				                             << ANSI_TXT_DFT << std::endl;
		}
		else
		{
			std::cout << COUT_GTEST_RED_SLOW << "slower by "
				                             << " = "
				                             << ((gpu.Elapsed() - cpu.Elapsed()) / cpu.Elapsed()) * 100
				                             << " %"
				                             << ANSI_TXT_DFT << std::endl;
		}


	}

	// compare
	inline bool TestCompare(const test::Matrix& mat1, const qlm::Matrix& mat2, const float threshold)
	{
		test::Matrix mat_gpu{ mat2.Rows(), mat2.Columns() };
		mat2.ToCPU(mat_gpu.data, mat_gpu.Rows(), mat_gpu.Columns());

		for (int r = 0; r < mat1.Rows(); r++)
		{
			for (int c = 0; c < mat1.Columns(); c++)
			{
				if (std::abs(mat1.Get(r, c) - mat_gpu.Get(r, c)) > threshold)
				{
					return false;
				}
			}
		}

		return true;
	}

	inline bool TestCompare(const test::Vector& vec1, const qlm::Vector& vec2, const float threshold)
	{
		test::Vector vec_gpu{ vec2.Length() };
		vec2.ToCPU(vec_gpu.data, vec_gpu.Length());

		for (int i = 0; i < vec1.Length(); i++)
		{
			if (std::abs(vec1.Get(i) - vec_gpu.Get(i)) > threshold)
			{
				return false;
			}
		}

		return true;
	}

	inline bool TestCompare(const float& src1, const float& src2, const float threshold)
	{
		if (std::abs(src1 - src2) > threshold)
		{
			return false;
		}

		return true;
	}

	inline bool TestCompare_Percentage(const qlm::Vector& vec1, const qlm::Vector& vec2, const float threshold)
	{
		for (int i = 0; i < vec1.Length(); i++)
		{
			if (!TestCompare_Percentage(vec1.Get(i), vec2.Get(i), threshold))
			{
				return false;
			}
		}

		return true;
	}

	inline bool TestCompare_Percentage(const float& src1, const float& src2, const float threshold)
	{
		const float percentage = (std::abs(src1 - src2) / (std::abs(src1 + src2) / 2.0f)) * 100.0f;
		
		if (percentage > threshold)
		{
			std::cout << "percentage = " << percentage 
				      << " ,threshold = " << threshold << "\n";

			return false;
		}

		return true;
	}
}