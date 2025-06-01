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

        public:
            Matrix();
            Matrix(const int rows, const int columns);
            Matrix(const Matrix& other);
            ~Matrix();

        public:
            void Set(const int r, const int c, const float value);
            void Set(const int i, const float value);
            float Get(const int r, const int c) const;
            float Get(const int i) const;
            int Columns() const;
            int Rows() const;

        public:
            // print matrix
            void Print() const;
            // random initialization
            void RandomInit(const float min_value, const float max_value);

        public:
            // matrix-matrix operations
            Status Add(const Matrix& src, Matrix& dst) const;
            Status Sub(const Matrix& src, Matrix& dst) const;
            Status Mul(const Matrix& src, Matrix& dst) const;
            Status Div(const Matrix& src, Matrix& dst) const;
            Status Dot(const Matrix& src, Matrix& dst) const;
            Status Transpose(Matrix& dst) const;

        public:
            // matrix-vector operations
            Status Add(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            Status Sub(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            Status Mul(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            Status Div(const Vector& src, Matrix& dst, const BroadCast& broad_cast) const;
            Status Dot(const Vector& src, Vector& dst) const;

        public:
            // matrix-scalar operations
            Status Add(const float src, Matrix& dst) const;
            Status Sub(const float src, Matrix& dst) const;
            Status Mul(const float src, Matrix& dst) const;
            Status Div(const float src, Matrix& dst) const;
    };
}