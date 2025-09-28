#pragma once
#include "vector.hpp"


namespace qlm
{
    // vector-vector operations
    void Add(const Vector& src0, const Vector& src1, Vector& dst);
    void Sub(const Vector& src0, const Vector& src1, Vector& dst);
    void Mul(const Vector& src0, const Vector& src1, Vector& dst);
    void Div(const Vector& src0, const Vector& src1, Vector& dst);
    void Cov(const Vector& src0, const Vector& src1, DeviceFloat& dst);
    void Corr(const Vector& src0, const Vector& src1, DeviceFloat& dst);
    void Dot(const Vector& src0, const Vector& src1, DeviceFloat& dst);
    void Angle(const Vector& src0, const Vector& src1, DeviceFloat& dst);

    // vector operations
    void Mag(const Vector& src, DeviceFloat& dst);
    void Unit(const Vector& src, Vector& dst);
    void Sum(const Vector& src, DeviceFloat& result);
    void Mean(const Vector& src, DeviceFloat& dst);
    void Var(const Vector& src, DeviceFloat& dst);
    void Min(const Vector& src, DeviceFloat& dst);
    void Max(const Vector& src, DeviceFloat& dst);
    void MinMax(const Vector& src, DeviceFloat& dst_min, DeviceFloat& dst_max);
    void Norm(const Vector& src, const Norm_t norm, DeviceFloat& dst);
    void ArgMin(const Vector& src, Vector& dst);
    void ArgMax(const Vector& src, Vector& dst);
    void ArgMinMax(const Vector& src, Vector& dst_min,Vector& dst_max);
    void WeightedSum(const Vector& src, const Vector& weights, const DeviceFloat bias, DeviceFloat& dst);
               
    // Vector-scalar operations
    void Add(const Vector& in, const DeviceFloat& val, Vector& dst);
    void Sub(const Vector& in, const DeviceFloat& val, Vector& dst);
    void Mul(const Vector& in, const DeviceFloat& val, Vector& dst);
    void Div(const Vector& in, const DeviceFloat& val, Vector& dst);
}