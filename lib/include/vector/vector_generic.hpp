#pragma once

#include "array.hpp"


namespace qlm
{
    // vector-vector operations
    template<MemType mem_type>
    void Add(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Sub(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Mul(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Div(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Cov(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void Corr(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void Dot(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void Angle(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Array<1, mem_type>& dst);

    // vector operations
    template<MemType mem_type>
    void Mag(const Vector<mem_type>& src, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void Unit(const Vector<mem_type>& src, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Sum(const Vector<mem_type>& src, Array<1, mem_type>& result);

    template<MemType mem_type>
    void Mean(const Vector<mem_type>& src, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void Var(const Vector<mem_type>& src, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void Min(const Vector<mem_type>& src, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void Max(const Vector<mem_type>& src, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void MinMax(const Vector<mem_type>& src, Array<1, mem_type>& dst_min, Array<1, mem_type>& dst_max);

    template<MemType mem_type>
    void Norm(const Vector<mem_type>& src, const Norm_t norm, Array<1, mem_type>& dst);

    template<MemType mem_type>
    void ArgMin(const Vector<mem_type>& src, Vector<mem_type>& dst);

    template<MemType mem_type>
    void ArgMax(const Vector<mem_type>& src, Vector<mem_type>& dst);

    template<MemType mem_type>
    void ArgMinMax(const Vector<mem_type>& src, Vector<mem_type>& dst_min, Vector<mem_type>& dst_max);

    template<MemType mem_type>
    void WeightedSum(const Vector<mem_type>& src, const Vector<mem_type>& weights, const Array<1, mem_type> bias, Array<1, mem_type>& dst);
               
    // Vector-scalar operations
    template<MemType mem_type>
    void Add(const Vector<mem_type>& in, const Array<1, mem_type>& val, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Sub(const Vector<mem_type>& in, const Array<1, mem_type>& val, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Mul(const Vector<mem_type>& in, const Array<1, mem_type>& val, Vector<mem_type>& dst);

    template<MemType mem_type>
    void Div(const Vector<mem_type>& in, const Array<1, mem_type>& val, Vector<mem_type>& dst);
}