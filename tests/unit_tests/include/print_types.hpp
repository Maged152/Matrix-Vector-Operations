#pragma once
#include <ostream>
#include "matrix_vector_op.hpp"

inline std::ostream& operator << (std::ostream& out, const qlm::Norm_t norm)
{
    switch (norm)
    {
    case qlm::Norm_t::L1_NORM :
        out << "L1_NORM";
        break;

    case qlm::Norm_t::L2_NORM:
        out << "L2_NORM";
        break;

    case qlm::Norm_t::INF_NORM:
        out << "INF_NORM";
        break;
    }

    return out;
}

inline std::ostream& operator << (std::ostream& out, const qlm::BroadCast bc)
{
    if (bc == qlm::BroadCast::BROAD_CAST_ROW)
    {
        out << "BROAD_CAST_ROW";
    }
    else
    {
        out << "BROAD_CAST_COLUMN";
    }

    return out;
}

inline std::ostream& operator << (std::ostream& out, const qlm::ConvMode mode)
{
    if (mode == qlm::ConvMode::FULL)
    {
        out << "FULL";
    }
    else if (mode == qlm::ConvMode::SAME)
    {
        out << "SAME";
    }
    else // mode == qlm::ConvMode::VALID
    {
        out << "VALID";
    }

    return out;
}