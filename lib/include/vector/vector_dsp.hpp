#pragma once
#include "vector.hpp"

namespace qlm
{
    void Conv(const Vector& input, const Vector& kernel, Vector& output, const qlm::ConvMode mode);
    
}