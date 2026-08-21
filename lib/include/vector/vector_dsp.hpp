#pragma once
#include "vector.hpp"

namespace qlm
{
    template<MemType mem_type>
    void Conv(const Vector<mem_type>& input, const Vector<mem_type>& kernel, Vector<mem_type>& output, const qlm::ConvMode mode);
    
}