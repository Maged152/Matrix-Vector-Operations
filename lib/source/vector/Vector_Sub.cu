#include "vector_processor/VectorProcessor_Elementwise.hpp"

namespace qlm
{
    template<MemType mem_type>
    void Sub(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst)
	{
        VectorProcessor_Elementwise<mem_type, OpSub>(src0, src1, dst);
	}

    template void Sub<MemType::MEM_GPU>(const Vector<MemType::MEM_GPU>&, const Vector<MemType::MEM_GPU>&, Vector<MemType::MEM_GPU>&);
    template void Sub<MemType::MEM_UM>(const Vector<MemType::MEM_UM>&, const Vector<MemType::MEM_UM>&, Vector<MemType::MEM_UM>&);
}