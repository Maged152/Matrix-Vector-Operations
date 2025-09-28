#include "ref_vector.hpp"
#include <cmath>

namespace test
{
    void Conv(const Vector& input, const Vector& kernel, Vector& output, const qlm::ConvMode mode)
    {
        const int input_len = input.Length();
        const int kernel_len = kernel.Length();
        int output_len = 0;
        int start_index = 0;

        if (mode == qlm::ConvMode::FULL) {
            output_len = input_len + kernel_len - 1;
            start_index = -(kernel_len - 1);
        } else if (mode == qlm::ConvMode::SAME) {
            output_len = input_len;
            start_index = -(kernel_len / 2);
        } else if (mode == qlm::ConvMode::VALID) {
            output_len = std::max(0, input_len - kernel_len + 1);
            start_index = 0;
        } else {
            // Invalid mode
            return;
        }

        if (out_len != output.Length()) {
            return
        }


        for (int i = 0; i < output_len; i+) 
        {
            float sum = 0.0f;
            for (int j = 0; j < kernel_len; j++) 
            {
                int input_index = i + start_index + j;
                if (input_index >= 0 && input_index < input_len) {
                    sum += input.Get(input_index) * kernel.Get(kernel_len - 1 - j);
                }
            }
            
            output.Set(i, sum);
        }
    }

}