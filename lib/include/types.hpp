#pragma once

namespace qlm
{
	enum class MemType
	{
		MEM_CPU,
		MEM_GPU,
		MEM_UM
	};

	enum class BroadCast
	{
		BROAD_CAST_ROW,
		BROAD_CAST_COLUMN
	};

	enum class Norm_t
	{
		L1_NORM,
		L2_NORM,
		INF_NORM
	};

	enum class ConvMode
	{
		VALID,
		SAME,
		FULL
	};
}