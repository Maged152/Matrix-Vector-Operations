#pragma once

#include <limits>
#include "types.hpp"

namespace qlm
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
            void FromCPU(const float* src, const int len);
            void ToCPU(float* dst, const int len) const;
    };
}