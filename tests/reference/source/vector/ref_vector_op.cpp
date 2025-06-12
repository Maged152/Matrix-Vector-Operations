#include "ref_vector.hpp"
#include <numbers>
#include <cmath>

namespace test
{
	void Vector::Add(const Vector& src, Vector& dst) const
	{
		for (int l = 0; l < src.Length(); l++)
		{
			const float res = this->Get(l) + src.Get(l);
			dst.Set(l, res);
		}
	}
	///////////////////////////////////////////////////////////////////////////
	void Vector::Sum(float& dst) const
	{
		dst = 0;
		for (int l = 0; l < this->Length(); l++)
		{
			dst += this->Get(l);
		}
	}
	/////////////////////////////////////////////////////////////////////////
	// void Angle(const test::Vector& src1, const test::Vector& src2, float& angle)
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
	// void ArgMax(const test::Vector& src, size_t& dst)
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
	// void ArgMin(const test::Vector& src, size_t& dst)
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
	// void ArgMinMax(const test::Vector& src, size_t& min, size_t& max)
	// {
	// 	ArgMin(src, min);
	// 	ArgMax(src, max);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Corr(const test::Vector& src1, const test::Vector& src2, float& dst)
	// {
	// 	float cov, var1, var2;

	// 	Cov(src1, src2, cov);
	// 	Var(src1, var1);
	// 	Var(src2, var2);

	// 	dst = cov / std::sqrt(var1 * var2);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Cov(const test::Vector& src1, const test::Vector& src2, float& dst)
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
	// void Div(const test::Vector& src1, const test::Vector& src2, test::Vector& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) / src2.Get(l);
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Dot(const test::Vector& src1, const test::Vector& src2, float& dst)
	// {
	// 	dst = 0;
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		dst += src1.Get(l) * src2.Get(l);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Mag(const test::Vector& src, float& dst)
	// {
	// 	dst = 0;
	// 	Dot(src, src, dst);
	// 	dst = std::sqrt(dst);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Max(const test::Vector& src, float& dst)
	// {
	// 	dst = src.Get(0);

	// 	for (int i = 1; i < src.Length(); i++)
	// 	{
	// 		dst = std::max(dst, src.Get(i));
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Mean(const test::Vector& src, float& dst)
	// {
	// 	Sum(src, dst);
	// 	dst /= src.Length();
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Min(const test::Vector& src, float& dst)
	// {
	// 	dst = src.Get(0);

	// 	for (int i = 1; i < src.Length(); i++)
	// 	{
	// 		dst = std::min(dst, src.Get(i));
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void MinMax(const test::Vector& src, float& min, float& max)
	// {
	// 	Min(src, min);
	// 	Max(src, max);
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Mul(const test::Vector& src1, const test::Vector& src2, test::Vector& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) * src2.Get(l);
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Norm(const test::Vector& src, qlm::Norm norm, float& dst)
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
	// void ScalarAdd(const test::Vector& src1, const float val, test::Vector& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) + val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ScalarDiv(const test::Vector& src1, const float val, test::Vector& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) / val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ScalarMul(const test::Vector& src1, const float val, test::Vector& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) * val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void ScalarSub(const test::Vector& src1, const float val, test::Vector& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) - val;
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Sub(const test::Vector& src1, const test::Vector& src2, test::Vector& dst)
	// {
	// 	for (int l = 0; l < src1.Length(); l++)
	// 	{
	// 		float res = src1.Get(l) - src2.Get(l);
	// 		dst.Set(l, res);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Sum(const test::Vector& src, float& dst)
	// {
	// 	dst = 0;
	// 	for (int l = 0; l < src.Length(); l++)
	// 	{
	// 		dst += src.Get(l);
	// 	}
	// }
	// ///////////////////////////////////////////////////////////////////////////
	// void Unit(const test::Vector& src, test::Vector& dst)
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
	// void Var(const test::Vector& src, float& dst)
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
	// void WeightedSum(const test::Vector& src, const test::Vector& weights, const float bias, float& dst)
	// {
	// 	Dot(src, weights, dst);
	// 	dst += bias;
	// }
	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////

}