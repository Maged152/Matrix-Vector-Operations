#pragma once

#include <limits>
#include "vector/array.hpp"

namespace test
{
    using VectorCPU = qlm::Vector<qlm::MemType::MEM_CPU>;
    using VectorUM = qlm::Vector<qlm::MemType::MEM_UM>;
    
    // vector-vector operations
    template<qlm::MemType mem_type>
    void Add(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst);

    template<qlm::MemType mem_type>
    void Sub(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst);

    template<qlm::MemType mem_type>
    void Mul(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst);

    template<qlm::MemType mem_type>
    void Div(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, qlm::Vector<mem_type>& dst);
    
    template<qlm::MemType mem_type>
    void Cov(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, float& dst);

    template<qlm::MemType mem_type>
    void Corr(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, float& dst);

    template<qlm::MemType mem_type>
    void Dot(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, float& dst);

    template<qlm::MemType mem_type>
    void Angle(const qlm::Vector<mem_type>& src0, const qlm::Vector<mem_type>& src1, float& dst);

    // vector operations
    template<qlm::MemType mem_type>
    void Mag(const qlm::Vector<mem_type>& src, float& dst);

    template<qlm::MemType mem_type>
    void Unit(const qlm::Vector<mem_type>& src, qlm::Vector<mem_type>& dst);

    template<qlm::MemType mem_type>
    void Sum(const qlm::Vector<mem_type>& src, float& result);

    template<qlm::MemType mem_type>
    void Mean(const qlm::Vector<mem_type>& src, float& dst);
    template<qlm::MemType mem_type>
    void Var(const qlm::Vector<mem_type>& src, float& dst);

    template<qlm::MemType mem_type>
    void Min(const qlm::Vector<mem_type>& src, float& dst);
    
    template<qlm::MemType mem_type>
    void Max(const qlm::Vector<mem_type>& src, float& dst);

    template<qlm::MemType mem_type>
    void MinMax(const qlm::Vector<mem_type>& src, float& dst_min, float& dst_max);

    template<qlm::MemType mem_type>
    void Norm(const qlm::Vector<mem_type>& src, const qlm::Norm_t norm, float& dst);

    template<qlm::MemType mem_type>
    void ArgMin(const qlm::Vector<mem_type>& src, VectorCPU& dst);

    template<qlm::MemType mem_type>
    void ArgMax(const qlm::Vector<mem_type>& src, VectorCPU& dst);

    template<qlm::MemType mem_type>
    void ArgMinMax(const qlm::Vector<mem_type>& src, VectorCPU& dst_min, VectorCPU& dst_max);

    template<qlm::MemType mem_type>
    void WeightedSum(const qlm::Vector<mem_type>& src, const qlm::Vector<mem_type>& weights, const float bias, float& dst);
               
    // Vector-scalar operations
    template<qlm::MemType mem_type>
    void Add(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst);

    template<qlm::MemType mem_type>
    void Sub(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst);

    template<qlm::MemType mem_type>
    void Mul(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst);
    
    template<qlm::MemType mem_type>
    void Div(const qlm::Vector<mem_type>& in, const float& val, qlm::Vector<mem_type>& dst);

    // vector dsp operations
    template<qlm::MemType mem_type>
    void Conv(const qlm::Vector<mem_type>& input, const qlm::Vector<mem_type>& kernel, qlm::Vector<mem_type>& output, const qlm::ConvMode mode);
}