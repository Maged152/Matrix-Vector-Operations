#include "types.hpp"

qlm::DeviceMemory::DeviceMemory(size_t size) : size(size)
{
    cudaMalloc(&data, size * sizeof(float));
}
qlm::DeviceMemory::~DeviceMemory()
{
    if (data != nullptr) {
        cudaFree(data);
        data = nullptr;
    }
}

void qlm::DeviceMemory::ToCPU(float *hostData) const
{
    cudaMemcpy(hostData, data, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void qlm::DeviceMemory::FromCPU(const float *hostData)
{
    cudaMemcpy(data, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
}