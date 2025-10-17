#include "ref_matrix.hpp"

namespace test
{
    void Dot(const Matrix& src0, const Matrix& src1, Matrix& dst)
    {
       for (int r = 0; r < src0.Rows(); r++)
        {
            for (int c = 0; c < src1.Columns(); c++)
            {
                float sum = 0.0f; // Initialize a variable to store the sum of products

                for (int e = 0; e < src0.Columns(); e++)
                {
                    sum += src0.Get(r, e) * src1.Get(e, c);
                }

                dst.Set(r, c, sum); // Set the result in the destination matrix
            }
        }
    }

}