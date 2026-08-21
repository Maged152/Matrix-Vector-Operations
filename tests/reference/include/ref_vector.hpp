#pragma once

#include <limits>
#include "vector/vector.hpp"

namespace test
{
    // CPU reference vector type alias
    using VectorCPU = qlm::Vector<qlm::MemType::MEM_CPU>;

    // vector-vector operations
    void Add(const VectorCPU& src0, const VectorCPU& src1, VectorCPU& dst);
    void Sub(const VectorCPU& src0, const VectorCPU& src1, VectorCPU& dst);
    void Mul(const VectorCPU& src0, const VectorCPU& src1, VectorCPU& dst);
    void Div(const VectorCPU& src0, const VectorCPU& src1, VectorCPU& dst);
    void Cov(const VectorCPU& src0, const VectorCPU& src1, float& dst);
    void Corr(const VectorCPU& src0, const VectorCPU& src1, float& dst);
    void Dot(const VectorCPU& src0, const VectorCPU& src1, float& dst);
    void Angle(const VectorCPU& src0, const VectorCPU& src1, float& dst);

    // vector operations
    void Mag(const VectorCPU& src, float& dst);
    void Unit(const VectorCPU& src, VectorCPU& dst);
    void Sum(const VectorCPU& src, float& result);
    void Mean(const VectorCPU& src, float& dst);
    void Var(const VectorCPU& src, float& dst);
    void Min(const VectorCPU& src, float& dst);
    void Max(const VectorCPU& src, float& dst);
    void MinMax(const VectorCPU& src, float& dst_min, float& dst_max);
    void Norm(const VectorCPU& src, const qlm::Norm_t norm, float& dst);
    void ArgMin(const VectorCPU& src, VectorCPU& dst);
    void ArgMax(const VectorCPU& src, VectorCPU& dst);
    void ArgMinMax(const VectorCPU& src, VectorCPU& dst_min,VectorCPU& dst_max);
    void WeightedSum(const VectorCPU& src, const VectorCPU& weights, const float bias, float& dst);
               
    // Vector-scalar operations
    void Add(const VectorCPU& in, const float& val, VectorCPU& dst);
    void Sub(const VectorCPU& in, const float& val, VectorCPU& dst);
    void Mul(const VectorCPU& in, const float& val, VectorCPU& dst);
    void Div(const VectorCPU& in, const float& val, VectorCPU& dst);

    // vector dsp operations
    void Conv(const VectorCPU& input, const VectorCPU& kernel, VectorCPU& output, const qlm::ConvMode mode);
}