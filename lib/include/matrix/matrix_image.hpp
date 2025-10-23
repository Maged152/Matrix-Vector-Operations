#pragma once

#include "matrix.hpp"

namespace qlm
{
    void Conv(const Matrix& input, const Matrix& kernel, Matrix& output, const qlm::ConvMode mode);
    
}