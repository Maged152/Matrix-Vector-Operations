#include "ref_matrix.hpp"
#include <stdexcept>
#include <cmath>

namespace test
{
    void Conv(const Matrix& src, const Matrix& kernel, Matrix& dst, const qlm::ConvMode conv_mode, const qlm::ConvMode mode)
    {
        const int ker_rows = kernel.rows;
        const int ker_cols = kernel.columns;
        const int center_x = ker_cols / 2;
        const int center_y = ker_rows / 2;

        const int src_rows = src.rows;
        const int src_cols = src.columns;

        int out_rows = 0;
        int out_cols = 0;
        int start_x = 0;
        int start_y = 0;

        // determine output size based on conv mode
        switch (mode)
        {
        case qlm::ConvMode::FULL:
            out_rows = src_rows + ker_rows - 1;
            out_cols = src_cols + ker_cols - 1;
            start_x = -(ker_cols - 1);
            start_y = -(ker_rows - 1);
            break;
        case qlm::ConvMode::SAME:
            out_rows = src_rows;
            out_cols = src_cols;
            start_x = -center_x;
            start_y = -center_y;
            break;
        case qlm::ConvMode::VALID:
            out_rows = src_rows - ker_rows + 1;
            out_cols = src_cols - ker_cols + 1;
            start_x = 0;
            start_y = 0;
            break;
        default:
            throw std::invalid_argument("Unsupported conv mode");
        }

        // validate dst size
        if (dst.rows != out_rows || dst.columns != out_cols)
            throw std::runtime_error("dst has incorrect dimensions for selected conv mode");

        auto out_of_bounds = [&](int r, int c)->bool {
            return (r < 0 || r >= src_rows || c < 0 || c >= src_cols);
        };

        for (int i = 0; i < out_rows; i++)
        {
            for (int j = 0; j < out_cols; j++)
            {
                float sum = 0.0f;

                for (int r = 0; r < ker_rows; ++r)
                {
                    for (int c = 0; c < ker_cols; ++c)
                    {
                        const int ii = i + c + start_y;
                        const int jj = j + c + start_x;

                        float pixel = 0.0f;
                        if (!out_of_bounds(ii, jj))
                        {
                            pixel = src.Get(ii, jj);
                        }

                        sum += pixel * kernel.Get(r, c);
                    }
                }

                dst.Set(i, j, sum);
            }
        }
    }

}