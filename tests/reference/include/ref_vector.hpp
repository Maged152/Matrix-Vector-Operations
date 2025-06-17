#pragma once

#include <limits>
#include "types.hpp"
namespace test
{
    class Vector 
    {
        public:
            float* data = nullptr;
            int length = 0;

        public:
            Vector();
            Vector(int length);
            Vector(const Vector& other);
            ~Vector();

        public:
            void Set(const int i, const float value);
            float Get(const int i) const;
            int Length() const;
            void Alloc(const int len);
        
        public:
            // print vector
            void Print();
            // random initialization
            void RandomInit(const float min_value, const float max_value);
    };


    // vector-vector operations
    void Add(const Vector& src0, const Vector& src1, Vector& dst);
    void Sub(const Vector& src0, const Vector& src1, Vector& dst);
    void Mul(const Vector& src0, const Vector& src1, Vector& dst);
    void Div(const Vector& src0, const Vector& src1, Vector& dst);
    void Cov(const Vector& src0, const Vector& src1, float& dst);
    void Corr(const Vector& src0, const Vector& src1, float& dst);
    void Dot(const Vector& src0, const Vector& src1, float& dst);
    void Angle(const Vector& src0, const Vector& src1, float& dst);

    // vector operations
    void Mag(const Vector& src, float& dst);
    void Unit(const Vector& src, Vector& dst);
    void Sum(const Vector& src, float& result);
    void Mean(const Vector& src, float& dst);
    void Var(const Vector& src, float& dst);
    void Min(const Vector& src, float& dst);
    void Max(const Vector& src, float& dst);
    void MinMax(const Vector& src, float& dst_min, float& dst_max);
    void Norm(const Vector& src, const Norm norm, float& dst);
    void ArgMin(const Vector& src, Vector& dst);
    void ArgMax(const Vector& src, Vector& dst);
    void ArgMinMax(const Vector& src, Vector& dst_min,Vector& dst_max);
    void WeightedSum(const Vector& src, const Vector& weights, const float bias, float& dst);
               
    // Vector-scalar operations
    void Add(const Vector& in, const float& val, Vector& dst);
    void Sub(const Vector& in, const float& val, Vector& dst);
    void Mul(const Vector& in, const float& val, Vector& dst);
    void Div(const Vector& in, const float& val, Vector& dst);
}