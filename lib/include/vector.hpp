#pragma once

#include <limits>
#include "types.hpp"

namespace qlm
{
    // class Matrix;

    class Vector 
    {
        private:
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
            void FromCPU(const float* src, const int len);
        
        public:
            // print vector
            void Print() const;
            // random initialization
            void RandomInit(const float min_value, const float max_value);

        public:
            // vector operations
            void Dot(const Vector& src, float& dst) const;
            void Mag(float& dst) const;
            void Unit(Vector& dst) const;
            void Angle(const Vector& src, float& dst) const;
            void Sum(float& dst) const;
            void Mean(float& dst) const;
            void Var(float& dst) const;
            void Min(float& dst) const;
            void Max(float& dst) const;
            void MinMax(float& dst_min, float& dst_max) const;
            void Norm(const Norm norm,float& dst) const;
            void ArgMin(size_t& dst) const;
            void ArgMax(size_t& dst) const;
            void ArgMinMax(size_t& dst_min, size_t& dst_max) const;
            void WeightedSum(const Vector& weights, const float bias, float& dst) const;

        public:
            // vector-vector operations
            void Add(const Vector& src, Vector& dst) const;
            void Sub(const Vector& src, Vector& dst) const;
            void Mul(const Vector& src, Vector& dst) const;
            void Div(const Vector& src, Vector& dst) const;
            void Cov(const Vector& src, float& dst) const;
            void Corr(const Vector& src, float& dst) const;
        public:
            // Vector-scalar operations
            void Add(const float src, Vector& dst) const;
            void Sub(const float src, Vector& dst) const;
            void Mul(const float src, Vector& dst) const;
            void Div(const float src, Vector& dst) const;

            //--------------------------------------------------------------//
            // friend class Matrix;
    };
}