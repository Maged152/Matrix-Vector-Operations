#pragma once

namespace qlm
{
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

	enum class BorderType
	{
		CONST,
		REPLICATE
	};

	struct BorderMode
	{
		BorderType border_type = BorderType::CONST;
		float border_pixel = 0.0f;
	};

	struct DeviceBuffer
    {
        float* data;
        size_t size = 0;

        DeviceBuffer(size_t size);
        ~DeviceBuffer();

        void ToCPU(float* host_data) const;
        void FromCPU(const float* host_data);
    };

	struct DeviceFloat
    {
		DeviceBuffer mem {1};
    };
}