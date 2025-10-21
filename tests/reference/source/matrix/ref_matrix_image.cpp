#include "ref_matrix.hpp"
#include <cmath>

namespace test
{
    void Conv(const Matrix& src0, const Matrix& kernel, Matrix& dst,  const qlm::BorderMode mode)
    {
        const int ker_rows = kernel.rows;
        const int ker_cols = kernel.columns;
        const int center_x = ker_cols / 2;
        const int center_y = ker_rows / 2;

        const int src_rows = src0.rows;
        const int src_cols = src0.columns;

        auto out_border = [&](const int r, const int c) -> bool
        {
            return (r < 0 || r >= src_rows || c < 0 || c >= src_cols);
        };

        for (int i = 0; i < src_rows; ++i)
        {
            for (int j = 0; j < src_cols; ++j)
            {
                float sum = 0.0f;

                for (int r = 0; r < ker_rows; ++r)
                {
                    for (int c = 0; c < ker_cols; ++c)
                    {
                        int ii = i + (r - center_y);
                        int jj = j + (c - center_x);

                        float pixel = 0.0f;

                        if (out_border(ii, jj))
                        {
                            if (mode.border_type == qlm::BorderType::CONST)
                            {
                                pixel = mode.border_pixel;
                            }
                            else if (mode.border_type == qlm::BorderType::REPLICATE)
                            {
                                ii = std::min(std::max(ii, 0), src_rows - 1);
                                jj = std::min(std::max(jj, 0), src_cols - 1);
                                pixel = src0.Get(ii, jj);
                            }
                        }
                        else
                        {
                            pixel = src0.Get(ii, jj);
                        }

                        sum += pixel * kernel.Get(r, c);
                    }
                }

                dst.Set(i, j, sum);
            }
        }
    }

}