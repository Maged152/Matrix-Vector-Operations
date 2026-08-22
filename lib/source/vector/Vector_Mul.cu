#include "vector_processor/VectorProcessor_Elementwise.hpp"

namespace qlm
{
    template<MemType mem_type>
    void Mul(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst)
	{
        VectorProcessor_Elementwise<mem_type, OpMul>(src0, src1, dst);
	}

    template void Mul<MemType::MEM_GPU>(const Vector<MemType::MEM_GPU>&, const Vector<MemType::MEM_GPU>&, Vector<MemType::MEM_GPU>&);
    template void Mul<MemType::MEM_UM>(const Vector<MemType::MEM_UM>&, const Vector<MemType::MEM_UM>&, Vector<MemType::MEM_UM>&);
}