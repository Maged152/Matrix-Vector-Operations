#include "ref_matrix.hpp"

namespace test
{
    void Matrix::Dot(const Matrix& src, Matrix& dst) const
    {
       for (int r = 0; r < this->Rows(); r++)
        {
            for (int c = 0; c < src.Columns(); c++)
            {
                float sum = 0.0f; // Initialize a variable to store the sum of products

                for (int e = 0; e < this->Columns(); e++)
                {
                    sum += this->Get(r, e) * src.Get(e, c);
                }

                dst.Set(r, c, sum); // Set the result in the destination matrix
            }
        }
    }

}