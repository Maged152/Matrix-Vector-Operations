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

    };


    // matrix-matrix operations
    void Add(const Matrix& src0, const Matrix& src1, Matrix& dst);
    void Sub(const Matrix& src0, const Matrix& src1, Matrix& dst);
    void Mul(const Matrix& src0, const Matrix& src1, Matrix& dst);
    void Div(const Matrix& src0, const Matrix& src1, Matrix& dst);
    void Dot(const Matrix& src0, const Matrix& src1, Matrix& dst);
    void Transpose(const Matrix& src, Matrix& dst);

    // matrix-vector operations
    void Add(const Matrix& src0, const Vector& src1, Matrix& dst, const qlm::BroadCast& broad_cast);
    void Sub(const Matrix& src0, const Vector& src1, Matrix& dst, const qlm::BroadCast& broad_cast);
    void Mul(const Matrix& src0, const Vector& src1, Matrix& dst, const qlm::BroadCast& broad_cast);
    void Div(const Matrix& src0, const Vector& src1, Matrix& dst, const qlm::BroadCast& broad_cast);
    void Dot(const Matrix& src0, const Vector& src1, Vector& dst);

    // matrix-scalar operations
    void Add(const Matrix& src0, const float src, Matrix& dst);
    void Sub(const Matrix& src0, const float src, Matrix& dst);
    void Mul(const Matrix& src0, const float src, Matrix& dst);
    void Div(const Matrix& src0, const float src, Matrix& dst);

    // image processing operation
    void Conv(const Matrix& src, const Matrix& kernel, Matrix& dst, const qlm::BorderMode mode);
}