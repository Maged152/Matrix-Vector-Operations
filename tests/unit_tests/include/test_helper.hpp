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

	inline bool TestCompare_SNR(const test::Vector& ref, const qlm::Vector& test, float snr_threshold_db)
	{
		test::Vector test_cpu{ test.Length() };
		test.ToCPU(test_cpu.data, test_cpu.Length());

		double signal_energy = 0.0;
		double noise_energy = 0.0;

		for (int i = 0; i < ref.Length(); ++i)
		{
			float ref_val = ref.Get(i);
			float test_val = test_cpu.Get(i);

			signal_energy += ref_val * ref_val;

			float noise = ref_val - test_val;
			noise_energy += noise * noise;
		}

		// Avoid division by zero
		if (noise_energy == 0.0)
		{
			test::PrintParameter(std::numeric_limits<double>::infinity(), "SNR (dB)");
			return true; // Perfect match
		}

		double snr_db = 10.0 * std::log10(signal_energy / noise_energy);
		test::PrintParameter(snr_db, "SNR (dB)");

		return snr_db >= snr_threshold_db;
	}

	// SNR compare for scalars
	inline bool TestCompare_SNR(float ref, float test, float snr_threshold_db)
	{
		double signal_energy = ref * ref;
		double noise_energy = (ref - test) * (ref - test);

		if (noise_energy == 0.0)
		{
			test::PrintParameter(std::numeric_limits<double>::infinity(), "SNR (dB)");
			return true; // Perfect match
		}

		double snr_db = 10.0 * std::log10(signal_energy / noise_energy);
		test::PrintParameter(snr_db, "SNR (dB)");

		return snr_db >= snr_threshold_db;
	}
}