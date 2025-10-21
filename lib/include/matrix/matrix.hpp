#pragma once

#include <limits>
#include "types.hpp"

namespace qlm
{
    class Vector;

    class Matrix 
    {
        public:
            float* data = nullptr;
            int columns = 0;
            int rows = 0;
            int stride = 0; 

        public:
            Matrix();
            Matrix(const int rows, const int columns, const int stride = -1);
            Matrix(const Matrix& other);
            ~Matrix();

        public:
            void Set(const int r, const int c, const float value);
            float Get(const int r, const int c) const;
            int Columns() const;
            int Rows() const;
            int Stride() const;
            void Alloc(const int rows, const int columns, const int stride = -1);
            void FromCPU(const float* src, const int rows, const int stride);
            void ToCPU(float* dst, const int rows, const int stride) const;

    };
    
    // matrix-matrix operations
    void Dot(const Matrix& src0, const Matrix& src1, Matrix& dst);
}