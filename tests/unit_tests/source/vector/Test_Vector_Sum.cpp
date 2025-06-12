#include "test_helper.hpp"
#include "matrix_vector_op.hpp"
#include "reference.hpp"


// Define the test parameters types
struct VectorSum : ::testing::TestWithParam<std::tuple<
    int,   // length
    float, // min value
    float  // max value
    >>
{};


// Define a parameterized test case
TEST_P(VectorSum, Test_VectorSum)
{
    constexpr float threshold = 100.0f;
    // extract the parameters
    auto& [length, min_val, max_val] = GetParam();

    // print the parameters
    test::PrintParameter(length, "length");
    test::PrintParameter(min_val, "min_val");
    test::PrintParameter(max_val, "max_val");

    qlm::Timer<qlm::usec> timer_cpu;
    qlm::Timer<qlm::usec> timer_gpu;

    float dst_cpu, dst_gpu;

    // cpu vector
    test::Vector src_cpu{ length };

    // gpu vectors
    qlm::Vector src_gpu{ length };

    // random initialization
    src_cpu.RandomInit(min_val, max_val);

    // copy to gpu
    src_gpu.FromCPU(src_cpu.data, length);

    // run cpu code
    timer_cpu.Start();
    src_cpu.Sum(dst_cpu);
    timer_cpu.End();

    // run gpu code
    timer_gpu.Start();
    src1_gpu.Sum(dst_gpu);
    timer_gpu.End();

    // print time
    test::PrintTime(timer_cpu, timer_gpu);

    // compare the results
    bool res = test::TestCompare_SNR(dst_cpu, dst_gpu, threshold);

    EXPECT_EQ(res, true);
}


// Instantiate the test case with combinations of values
INSTANTIATE_TEST_CASE_P(
    Test_VectorSum, VectorSum,
    ::testing::Combine(
        ::testing::Values(7, 100, 5000, 20000, 200000, 2000000),
        ::testing::Values(0.0f, -100.0f),
        ::testing::Values(1.0f, 100.0f)
    ));