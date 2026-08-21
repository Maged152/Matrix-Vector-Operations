#pragma once

#include "vector.hpp"

namespace qlm
{
    template<int N, MemType mem_type>
    class Array
    {
        static_assert(N > 0, "Array size must be positive");

    private:
        qlm::Vector<mem_type> vec{ N };

    public:
        Array() = default;
        Array(const Array& other) = default;

        static constexpr int Length() { return N; }

        float* Data() { return vec.Data(); }
        const float* Data() const { return vec.Data(); }

        void Set(const int i, const float value) { vec.Set(i, value); }
        float Get(const int i) const { return vec.Get(i); }

        void FromCPU(const float* src) { vec.FromCPU(src, N); }
        void ToCPU(float* dst) const { vec.ToCPU(dst, N); }

        void Print() const { vec.Print(); }
        void RandomInit(const float min_value, const float max_value) { vec.RandomInit(min_value, max_value); }
    };
}