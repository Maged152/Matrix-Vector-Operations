#include "test_helper.hpp"
#include "matrix_vector_op.hpp"
#include "reference.hpp"

struct VectorConv : ::testing::TestWithParam<std::tuple<
    int,   // input length
    int,   // kernel length
    float, // min value
    float, // max value
    qlm::ConvMode // convolution mode
    >>
{};

TEST_P(VectorConv, Test_VectorConv)
{
    constexpr float snr_threshold_db = 100.0f;
    auto& [input_length, kernel_length, min_val, max_val, mode] = GetParam();

    test::PrintParameter(input_length, "input_length");
    test::PrintParameter(kernel_length, "kernel_length");
    test::PrintParameter(min_val, "min_val");
    test::PrintParameter(max_val, "max_val");
    test::PrintParameter(static_cast<int>(mode), "conv_mode");

    // Determine output length
    int output_length = 0;
    if (mode == qlm::ConvMode::FULL)
        output_length = input_length + kernel_length - 1;
    else if (mode == qlm::ConvMode::SAME)
        output_length = input_length;
    else if (mode == qlm::ConvMode::VALID)
        output_length = std::max(0, input_length - kernel_length + 1);

    // CPU vectors
    test::Vector input_cpu{ input_length };
    test::Vector kernel_cpu{ kernel_length };
    test::Vector output_cpu{ output_length };

    // GPU vectors
    qlm::Vector input_gpu{ input_length };
    qlm::Vector kernel_gpu{ kernel_length };
    qlm::Vector output_gpu{ output_length };

    // Random initialization
    input_cpu.RandomInit(min_val, max_val);
    kernel_cpu.RandomInit(min_val, max_val);

    qlm::Timer<qlm::usec> timer_cpu;
    qlm::Timer<qlm::usec> timer_gpu;

    // Copy to GPU
    input_gpu.FromCPU(input_cpu.data, input_length);
    kernel_gpu.FromCPU(kernel_cpu.data, kernel_length);

    // Run CPU code
    timer_cpu.Start();
    test::Conv(input_cpu, kernel_cpu, output_cpu, mode);
    timer_cpu.End();

    // Run GPU code
    timer_gpu.Start();
    qlm::Conv(input_gpu, kernel_gpu, output_gpu, mode);
    timer_gpu.End();

    // Print time
    test::PrintTime(timer_cpu, timer_gpu);

    // Compare results using SNR
    bool res = test::TestCompare_SNR(output_cpu, output_gpu, snr_threshold_db);
    EXPECT_EQ(res, true);
}

INSTANTIATE_TEST_CASE_P(
    Test_VectorConv, VectorConv,
    ::testing::Combine(
        ::testing::Values(8, 32, 100, 256, 1024, 5000),
        ::testing::Values(3, 5, 13, 27, 51),
        ::testing::Values(-1.0f, 0.0f),
        ::testing::Values(1.0f, 10.0f),
        ::testing::Values(
            qlm::ConvMode::FULL,
            qlm::ConvMode::SAME,
            qlm::ConvMode::VALID
        )
    ));
