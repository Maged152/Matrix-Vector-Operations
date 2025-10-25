#include "test_helper.hpp"
#include "matrix_vector_op.hpp"
#include "reference.hpp"
#include "print_types.hpp"

struct MatrixConv : ::testing::TestWithParam<std::tuple<
    int, // src rows
    int, // src cols
    int, // ker rows
    int, // ker cols
    float, // min val
    float, // max val
    qlm::ConvMode
    >>
{};

TEST_P(MatrixConv, Test_MatrixConv)
{
    constexpr float snr_threshold_db = 100.0f;
    auto& [src_rows, src_cols, ker_rows, ker_cols, min_val, max_val, mode] = GetParam();

    test::PrintParameter(src_rows, "src_rows");
    test::PrintParameter(src_cols, "src_cols");
    test::PrintParameter(ker_rows, "ker_rows");
    test::PrintParameter(ker_cols, "ker_cols");
    test::PrintParameter(min_val, "min_val");
    test::PrintParameter(max_val, "max_val");
    test::PrintParameter(mode, "conv_mode");

    int dst_rows = 0;
    int dst_cols = 0;
    if (mode == qlm::ConvMode::FULL) {
        dst_rows = src_rows + ker_rows - 1;
        dst_cols = src_cols + ker_cols - 1;
    } else if (mode == qlm::ConvMode::SAME) {
        dst_rows = src_rows;
        dst_cols = src_cols;
    } else { // VALID
        dst_rows = std::max(0, src_rows - ker_rows + 1);
        dst_cols = std::max(0, src_cols - ker_cols + 1);
    }

    // CPU matrices
    test::Matrix src_cpu{ src_rows, src_cols };
    test::Matrix kernel_cpu{ ker_rows, ker_cols };
    test::Matrix dst_cpu{ dst_rows, dst_cols };

    // GPU matrices
    qlm::Matrix src_gpu{ src_rows, src_cols };
    qlm::Matrix kernel_gpu{ ker_rows, ker_cols };
    qlm::Matrix dst_gpu{ dst_rows, dst_cols };

    // initialize
    src_cpu.RandomInit(min_val, max_val);
    kernel_cpu.RandomInit(min_val, max_val);

    // copy to gpu
    src_gpu.FromCPU(src_cpu.data, src_rows, src_cols);
    kernel_gpu.FromCPU(kernel_cpu.data, ker_rows, ker_cols);

    qlm::Timer<qlm::usec> timer_cpu;
    qlm::Timer<qlm::usec> timer_gpu;

    // run cpu reference
    timer_cpu.Start();
    test::Conv(src_cpu, kernel_cpu, dst_cpu, mode);
    timer_cpu.End();

    // run gpu
    timer_gpu.Start();
    qlm::Conv(src_gpu, kernel_gpu, dst_gpu, mode);
    timer_gpu.End();

    test::PrintTime(timer_cpu, timer_gpu);

    bool res = test::TestCompare_SNR(dst_cpu, dst_gpu, snr_threshold_db);
    EXPECT_EQ(res, true);
}

INSTANTIATE_TEST_CASE_P(
    Test_MatrixConv, MatrixConv,
    ::testing::Combine(
        ::testing::Values(16, 32, 64),
        ::testing::Values(16, 32, 48),
        ::testing::Values(3, 5),
        ::testing::Values(3, 5),
        ::testing::Values(-1.0f),
        ::testing::Values(1.0f),
        ::testing::Values(
            qlm::ConvMode::FULL,
            qlm::ConvMode::SAME,
            qlm::ConvMode::VALID
        )
    )
);
