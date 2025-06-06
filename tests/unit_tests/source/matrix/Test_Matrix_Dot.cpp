#include "test_helper.hpp"
#include "matrix_vector_op.hpp"
#include "reference.hpp"


// Define the test parameters types
struct MatrixDot : ::testing::TestWithParam<std::tuple<
    int,   // rows
    int,   // cols
    float, // min value
    float  // max value
    >>
{};


// Define a parameterized test case
TEST_P(MatrixDot, Test_MatrixDot)
{
    constexpr float threshold = 0.0f;
    // extract the parameters
    auto& [rows, cols, min_val, max_val] = GetParam();

    // print the parameters
    test::PrintParameter(rows, "rows");
    test::PrintParameter(cols, "cols");
    test::PrintParameter(min_val, "min_val");
    test::PrintParameter(max_val, "max_val");

    qlm::Timer<qlm::usec> timer_cpu;
    qlm::Timer<qlm::usec> timer_gpu;

    // cpu Matrices
    test::Matrix src1_cpu{ rows, cols };
    test::Matrix src2_cpu{ cols, rows };
    test::Matrix dst_cpu{ rows, rows };

    // gpu Matrices
    qlm::Matrix src1_gpu{ rows, cols };
    qlm::Matrix src2_gpu{ cols, rows };
    qlm::Matrix dst_gpu{ rows, rows };

    // random initialization
    src1_cpu.RandomInit(min_val, max_val);
    src2_cpu.RandomInit(min_val, max_val);

    // copy to gpu
    src1_gpu.FromCPU(src1_cpu.data, rows, cols);
    src2_gpu.FromCPU(src2_cpu.data, cols, rows);

    // run cpu code
    timer_cpu.Start();
    src1_cpu.Dot(src2_cpu, dst_cpu);
    timer_cpu.End();

    // run gpu code
    timer_gpu.Start();
    src1_gpu.Dot(src2_gpu, dst_gpu);
    timer_gpu.End();

    // print time
    test::PrintTime(timer_cpu, timer_gpu);

    // compare the results
    bool res = test::TestCompare(dst_cpu, dst_gpu, threshold);

    EXPECT_EQ(res, true);
}


// Instantiate the test case with combinations of values
INSTANTIATE_TEST_CASE_P(
    Test_MatrixDot, MatrixDot,
    ::testing::Combine(
        ::testing::Values(16),//7, 100, 5000, 20000, 200000),
        ::testing::Values(16),//7, 100, 5000, 20000, 200000),
        ::testing::Values(0.0f),
        ::testing::Values(100.0f)
    ));