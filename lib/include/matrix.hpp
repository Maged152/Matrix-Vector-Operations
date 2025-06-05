#pragma once

#include <limits>
#include "types.hpp"

namespace qlm
{
    class Vector;

    class Matrix 
    {
        private:
            float* data = nullptr;
            int columns = 0;
            int rows = 0;
            size_t stride = 0; 

        public:
            Matrix();
            Matrix(const int rows, const int columns);
            Matrix(const Matrix& other);
            ~Matrix();

        public:
            void Set(const int r, const int c, const float value);
            float Get(const int r, const int c) const;
            int Columns() const;
            int Rows() const;
            int Stride() const;
            void Alloc(const int rows, const int columns);
            void FromCPU(const float* src, const int rows, const int columns);
            void ToCPU(float* dst, const int rows, const int columns) const;

        public:
            // matrix-matrix operations
            void Add(const Matrix& src, Matrix& dst) const;
            void Sub(const Matrix& src, Matrix& dst) const;
            void Mul(const Matrix& src, Matrix& dst) const;
            void Div(const Matrix& src, Matrix& dst) const;
            void Dot(const Matrix& src, Matrix& dst) const;
            void Transpose(Matrix& dst) const;

        public:
            // matrix-vector operations
            void Add(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            void Sub(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            void Mul(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            void Div(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            void Dot(const Vector& src, Vector& dst) const;

        public:
            // matrix-scalar operations
            void Add(const float src, Matrix& dst) const;
            void Sub(const float src, Matrix& dst) const;
            void Mul(const float src, Matrix& dst) const;
            void Div(const float src, Matrix& dst) const;
    };
}