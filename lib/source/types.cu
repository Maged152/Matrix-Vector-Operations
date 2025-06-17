#include "types.hpp"

qlm::DeviceBuffer::DeviceBuffer(size_t size) : size(size)
{
    cudaMalloc(&data, size * sizeof(float));
}
qlm::DeviceBuffer::~DeviceBuffer()
{
    if (data != nullptr) {
        cudaFree(data);
        data = nullptr;
    }
}

void qlm::DeviceBuffer::ToCPU(float *hostData) const
{
    cudaMemcpy(hostData, data, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void qlm::DeviceBuffer::FromCPU(const float *hostData)
{
    cudaMemcpy(data, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
}