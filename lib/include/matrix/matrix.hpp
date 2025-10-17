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

    };
    
    // matrix-matrix operations
    void Dot(const Matrix& src0, const Matrix& src1, Matrix& dst);
}