#pragma once

#include <limits>
#include <cstring>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "types.hpp"

namespace qlm
{
    template<MemType mem_type>
    class Vector 
    {
        private:
            float* data = nullptr;
            int length = 0;

            void Release()
            {
                if (data != nullptr)
                {
                    if constexpr (mem_type == MemType::MEM_CPU)
                    {
                        delete[] data;
                    }
                    else // MEM_GPU or MEM_UM
                    {
                        cudaFree(data);
                    }
                    data = nullptr;
                }
            }

        public:
            Vector() = default;
            Vector(int length) : length(length)
            {
                Alloc(length);
            }
            Vector(const Vector& other) : length(other.length)
            {
                Alloc(length);
                if constexpr (mem_type == MemType::MEM_GPU)
                {
                    cudaMemcpy(data, other.data, length * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                else // MEM_UM or MEM_CPU
                {
                    std::memcpy(data, other.data, length * sizeof(float));
                }
            }
            ~Vector()
            {
                Release();
            }

        public:
            float* Data() { return data; }
            const float* Data() const { return data; }

            void Set(const int i, const float value)
            {
                if (i >= 0 && i < length)
                {
                    if constexpr (mem_type == MemType::MEM_GPU)
                    {
                        cudaMemcpy(&data[i], &value, sizeof(float), cudaMemcpyHostToDevice);
                    }
                    else
                    {
                        data[i] = value;
                    }
                }
            }

            float Get(const int i) const
            {
                float value = std::numeric_limits<float>::signaling_NaN();
                if (i >= 0 && i < length)
                {
                    if constexpr (mem_type == MemType::MEM_GPU)
                    {
                        cudaMemcpy(&value, &data[i], sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    else
                    {
                        value = data[i];
                    }
                }
                return value;
            }

            int Length() const
            {
                return length;
            }

            void Alloc(const int len)
            {
                Release();
                length = len;
                if (length > 0)
                {
                    if constexpr (mem_type == MemType::MEM_CPU)
                    {
                        data = new float[length];
                    }
                    else if constexpr (mem_type == MemType::MEM_GPU)
                    {
                        cudaMalloc(&data, length * sizeof(float));
                    }
                    else // MEM_UM
                    {
                        cudaMallocManaged(&data, length * sizeof(float));
                    }
                }
            }

            void FromCPU(const float* src, const int len)
            {
                Alloc(len);
                if (data != nullptr && src != nullptr)
                {
                    if constexpr (mem_type == MemType::MEM_GPU)
                    {
                        cudaMemcpy(data, src, length * sizeof(float), cudaMemcpyHostToDevice);
                    }
                    else // MEM_UM or MEM_CPU
                    {
                        std::memcpy(data, src, length * sizeof(float));
                    }
                }
            }

            void ToCPU(float* dst, const int len) const
            {
                if (data != nullptr && dst != nullptr && len == length)
                {
                    if constexpr (mem_type == MemType::MEM_GPU)
                    {
                        cudaMemcpy(dst, data, length * sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    else // MEM_UM or MEM_CPU
                    {
                        std::memcpy(dst, data, length * sizeof(float));
                    }
                }
            }

            void PrefetchToGPU() const
            {
                if constexpr (mem_type == MemType::MEM_UM)
                {
                    if (data != nullptr && length > 0)
                    {
                        int device_id = 0;
                        cudaGetDevice(&device_id);
                        cudaMemPrefetchAsync(data, length * sizeof(float), device_id);
                    }
                }
            }

            void PrefetchToCPU() const
            {
                if constexpr (mem_type == MemType::MEM_UM)
                {
                    if (data != nullptr && length > 0)
                    {
                        cudaMemPrefetchAsync(data, length * sizeof(float), cudaCpuDeviceId);
                    }
                }
            }

            // print vector
            void Print() const
            {
                int number_digits = 5;

                for (int l = 0; l < length; l++)
                {
                    float element = this->Get(l);

                    if (element != 0)
                    {
                        int digits = static_cast<int>(std::log10(std::abs(element))) + 1;
                        number_digits = digits >= 5 ? 0 : 5 - digits;
                    }

                    std::cout << std::fixed << std::setprecision(number_digits) << element << " ";
                }
            }

            // random initialization
            void RandomInit(const float min_value, const float max_value)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dis(min_value, max_value);

                for (int i = 0; i < length; i++)
                {
                    this->Set(i, dis(gen));
                }
            }
    };
}