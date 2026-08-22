#include "ref_vector.hpp"
#include <numbers>
#include <cmath>

namespace test
{
	template<qlm::MemType mem_type>
	void test::Add(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst)
	{
		const int len =std::min(src0.Length(), src1.Length());
		for (int l = 0; l < len; l++)
		{
			const float res = src0.Get(l) + src1.Get(l);
			dst.Set(l, res);
		}
	}

	template void test::Add<qlm::MemType::MEM_CPU>(const VectorCPU&, const VectorCPU&, VectorCPU&);	
	template void test::Add<qlm::MemType::MEM_UM>(const VectorUM&, const VectorUM&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Sub(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst)
	{
		const int len =std::min(src0.Length(), src1.Length());
		for (int l = 0; l < len; l++)
		{
			const float res = src0.Get(l) - src1.Get(l);
			dst.Set(l, res);
		}
	}

	template void test::Sub<qlm::MemType::MEM_CPU>(const VectorCPU&, const VectorCPU&, VectorCPU&);	
	template void test::Sub<qlm::MemType::MEM_UM>(const VectorUM&, const VectorUM&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Mul(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst)
	{
		const int len =std::min(src0.Length(), src1.Length());
		for (int l = 0; l < len; l++)
		{
			const float res = src0.Get(l) * src1.Get(l);
			dst.Set(l, res);
		}
	}

	template void test::Mul<qlm::MemType::MEM_CPU>(const VectorCPU&, const VectorCPU&, VectorCPU&);	
	template void test::Mul<qlm::MemType::MEM_UM>(const VectorUM&, const VectorUM&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Div(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst)
	{
		const int len =std::min(src0.Length(), src1.Length());
		for (int l = 0; l < len; l++)
		{
			const float res = src0.Get(l) / src1.Get(l);
			dst.Set(l, res);
		}
	}

	template void test::Div<qlm::MemType::MEM_CPU>(const VectorCPU&, const VectorCPU&, VectorCPU&);	
	template void test::Div<qlm::MemType::MEM_UM>(const VectorUM&, const VectorUM&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Add(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst)
	{
		for (int l = 0; l < in.Length(); l++)
		{
			const float res = in.Get(l) + val;
			dst.Set(l, res);
		}
	}

	template void test::Add<qlm::MemType::MEM_CPU>(const VectorCPU&, const float&, VectorCPU&);	
	template void test::Add<qlm::MemType::MEM_UM>(const VectorUM&, const float&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Sub(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst)
	{
		for (int l = 0; l < in.Length(); l++)
		{
			const float res = in.Get(l) - val;
			dst.Set(l, res);
		}
	}

	template void test::Sub<qlm::MemType::MEM_CPU>(const VectorCPU&, const float&, VectorCPU&);	
	template void test::Sub<qlm::MemType::MEM_UM>(const VectorUM&, const float&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Mul(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst)
	{
		for (int l = 0; l < in.Length(); l++)
		{
			const float res = in.Get(l) * val;
			dst.Set(l, res);
		}
	}

	template void test::Mul<qlm::MemType::MEM_CPU>(const VectorCPU&, const float&, VectorCPU&);	
	template void test::Mul<qlm::MemType::MEM_UM>(const VectorUM&, const float&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Div(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst)
	{
		for (int l = 0; l < in.Length(); l++)
		{
			const float res = in.Get(l) / val;
			dst.Set(l, res);
		}
	}

	template void test::Div<qlm::MemType::MEM_CPU>(const VectorCPU&, const float&, VectorCPU&);	
	template void test::Div<qlm::MemType::MEM_UM>(const VectorUM&, const float&, VectorUM&);
	///////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Sum(const qlm::Vector<mem_type>& src, float& dst)
	{
		dst = 0;
		for (int l = 0; l < src.Length(); l++)
		{
			dst += src.Get(l);
		}
	}

	template void test::Sum<qlm::MemType::MEM_CPU>(const VectorCPU&, float&);	
	template void test::Sum<qlm::MemType::MEM_UM>(const VectorUM&, float&);
	/////////////////////////////////////////////////////////////////////////
	template<qlm::MemType mem_type>
	void test::Mean(const qlm::Vector<mem_type>& src, float& dst)
	{
		Sum(src, dst);
			dst /= src.Length();
	}

	template void test::Mean<qlm::MemType::MEM_CPU>(const VectorCPU&, float&);	
	template void test::Mean<qlm::MemType::MEM_UM>(const VectorUM&, float&);
	// ///////////////////////////////////////////////////////////////////////////
	// void Angle(const VectorCPU& src1, const VectorCPU& src2, float& angle)
	// {
	// 	// mag for src1
	// 	float mag1 = 0;
	// 	Mag(src1, mag1);
	// 	// mag for src2
	// 	float mag2 = 0;
	// 	Mag(src2, mag2);
	// 	// dot product
	// 	float dot = 0;
	// 	Dot(src1, src2, dot);

	// 	angle = std::acos(dot / (mag1 * mag2)) * 180.0f / std::numbers::pi;
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ArgMax(const VectorCPU& src, size_t& dst)
	// {
	// 	float max_val = src.Get(0);
	// 	dst = 0;

	// 	for (size_t i = 1; i < src.Length(); i++)
	// 	{
	// 		if (src.Get(i) > max_val)
	// 		{
	// 			max_val = src.Get(i);
	// 			dst = i;
	// 		}
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ArgMin(const VectorCPU& src, size_t& dst)
	// {
	// 	float min_val = src.Get(0);
	// 	dst = 0;

	// 	for (size_t i = 1; i < src.Length(); i++)
	// 	{
	// 		if (src.Get(i) < min_val)
	// 		{
	// 			min_val = src.Get(i);
	// 			dst = i;
	// 		}
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ArgMinMax(const VectorCPU& src, size_t& min, size_t& max)
	// {
	// 	ArgMin(src, min);
	// 	ArgMax(src, max);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Corr(const VectorCPU& src1, const VectorCPU& src2, float& dst)
	// {
	// 	float cov, var1, var2;

	// 	Cov(src1, src2, cov);
	// 	Var(src1, var1);
	// 	Var(src2, var2);

	// 	dst = cov / std::sqrt(var1 * var2);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Cov(const VectorCPU& src1, const VectorCPU& src2, float& dst)
	// {
	// 	float mean1, mean2;
	// 	Mean(src1, mean1);
	// 	Mean(src2, mean2);

	// 	dst = 0;
	// 	for (int i = 0; i < src1.Length(); i++)
	// 	{
	// 		dst += (src1.Get(i) - mean1) * (src2.Get(i) - mean2);
	// 	}

	// 	dst = dst / (src1.Length() - 1);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Div(const VectorCPU& src1, const VectorCPU& src2, VectorCPU& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) / src2.Get(l);
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Dot(const VectorCPU& src1, const VectorCPU& src2, float& dst)
	// {
	// 	dst = 0;
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		dst += src1.Get(l) * src2.Get(l);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Mag(const VectorCPU& src, float& dst)
	// {
	// 	dst = 0;
	// 	Dot(src, src, dst);
	// 	dst = std::sqrt(dst);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Max(const VectorCPU& src, float& dst)
	// {
	// 	dst = src.Get(0);

	// 	for (int i = 1; i < src.Length(); i++)
	// 	{
	// 		dst = std::max(dst, src.Get(i));
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Min(const VectorCPU& src, float& dst)
	// {
	// 	dst = src.Get(0);

	// 	for (int i = 1; i < src.Length(); i++)
	// 	{
	// 		dst = std::min(dst, src.Get(i));
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void MinMax(const VectorCPU& src, float& min, float& max)
	// {
	// 	Min(src, min);
	// 	Max(src, max);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Mul(const VectorCPU& src1, const VectorCPU& src2, VectorCPU& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) * src2.Get(l);
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Norm(const VectorCPU& src, qlm::Norm norm, float& dst)
	// {
	// 	if (norm == qlm::Norm::L1_NORM)
	// 	{
	// 		dst = 0;
	// 		for (int l = 0; l < src.Length(); l++)
	// 		{
	// 			dst += std::abs(src.Get(l));
	// 		}
	// 	}
	// 	else if (norm == qlm::Norm::L2_NORM)
	// 	{
	// 		Mag(src, dst);
	// 	}
	// 	else
	// 	{
	// 		Max(src, dst);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ScalarAdd(const VectorCPU& src1, const float val, VectorCPU& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) + val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ScalarDiv(const VectorCPU& src1, const float val, VectorCPU& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) / val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ScalarMul(const VectorCPU& src1, const float val, VectorCPU& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) * val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ScalarSub(const VectorCPU& src1, const float val, VectorCPU& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) - val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Sub(const VectorCPU& src1, const VectorCPU& src2, VectorCPU& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) - src2.Get(l);
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Sum(const VectorCPU& src, float& dst)
	// {
	// 	dst = 0;
	// 	for (int l = 0; l < src.Length(); l++)
	// 	{
	// 		dst += src.Get(l);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Unit(const VectorCPU& src, VectorCPU& dst)
	// {
	// 	float mag = 0;
	// 	Mag(src, mag);

	// 	for (int l = 0; l < src.Length(); l++)
	// 	{
	// 		float res = src.Get(l) / mag;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Var(const VectorCPU& src, float& dst)
	// {
	// 	float mean;
	// 	Mean(src, mean);
	// 	dst = 0;
	// 	for (int l = 0; l < src.Length(); l++)
	// 	{
	// 		dst += std::pow(src.Get(l) - mean, 2);
	// 	}

	// 	dst = dst / (src.Length() - 1);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void WeightedSum(const VectorCPU& src, const VectorCPU& weights, const float bias, float& dst)
	// {
	// 	Dot(src, weights, dst);
	// 	dst += bias;
	// }
	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////

}