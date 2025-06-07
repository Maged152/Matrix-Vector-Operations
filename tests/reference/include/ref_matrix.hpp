#pragma once

#include <limits>
#include "types.hpp"

namespace test
{
    class Vector;

    class Matrix 
    {
        public:
            float* data = nullptr;
            int columns = 0;
            int rows = 0;

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

        public:
            // print matrix
            void Print() const;
            // random initialization
            void RandomInit(const float min_value, const float max_value);
            void LinearInit();

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
            void Add(const Vector& src, Matrix& dst, const qlm::BroadCast& broad_cast) const;
            void Sub(const Vector& src, Matrix& dst, const qlm::BroadCast& broad_cast) const;
            void Mul(const Vector& src, Matrix& dst, const qlm::BroadCast& broad_cast) const;
            void Div(const Vector& src, Matrix& dst, const qlm::BroadCast& broad_cast) const;
            void Dot(const Vector& src, Vector& dst) const;

        public:
            // matrix-scalar operations
            void Add(const float src, Matrix& dst) const;
            void Sub(const float src, Matrix& dst) const;
            void Mul(const float src, Matrix& dst) const;
            void Div(const float src, Matrix& dst) const;
    };
}