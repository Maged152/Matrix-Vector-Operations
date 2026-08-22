#include "vector_processor/VectorProcessor_Elementwise.hpp"

namespace qlm
{
    template<MemType mem_type>
    void Add(const Vector<mem_type>& src0, const Vector<mem_type>& src1, Vector<mem_type>& dst)
	{
        VectorProcessor_Elementwise<mem_type, OpAdd>(src0, src1, dst);
	}

    template void Add<MemType::MEM_GPU>(const Vector<MemType::MEM_GPU>&, const Vector<MemType::MEM_GPU>&, Vector<MemType::MEM_GPU>&);
    template void Add<MemType::MEM_UM>(const Vector<MemType::MEM_UM>&, const Vector<MemType::MEM_UM>&, Vector<MemType::MEM_UM>&);

    template<MemType mem_type>
    void Add(const Vector<mem_type>& in, const Array<1, mem_type>& val, Vector<mem_type>& dst)
	{
        VectorProcessor_Elementwise<mem_type, OpAdd>(in, val, dst);
	}

    template void Add<MemType::MEM_GPU>(const Vector<MemType::MEM_GPU>&, const Array<1, MemType::MEM_GPU>&, Vector<MemType::MEM_GPU>&);
    template void Add<MemType::MEM_UM>(const Vector<MemType::MEM_UM>&, const Array<1, MemType::MEM_UM>&, Vector<MemType::MEM_UM>&);
}