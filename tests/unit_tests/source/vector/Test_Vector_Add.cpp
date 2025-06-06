#include "test_helper.hpp"
#include "matrix_vector_op.hpp"
#include "reference.hpp"


// Define the test parameters types
struct VectorAdd : ::testing::TestWithParam<std::tuple<
    int,   // length
    float, // min value
    float  // max value
    >>
{};


// Define a parameterized test case
TEST_P(VectorAdd, Test_VectorAdd)
{
    constexpr float threshold = 0.0f;
    // extract the parameters
    auto& [length, min_val, max_val] = GetParam();

    // print the parameters
    test::PrintParameter(length, "length");
    test::PrintParameter(min_val, "min_val");
    test::PrintParameter(max_val, "max_val");

    qlm::Timer<qlm::usec> timer_cpu;
    qlm::Timer<qlm::usec> timer_gpu;

    // cpu vectors
    test::Vector src1_cpu{ length };
    test::Vector src2_cpu{ length };
    test::Vector dst_cpu{ length };

    // gpu vectors
    qlm::Vector src1_gpu{ length };
    qlm::Vector src2_gpu{ length };
    qlm::Vector dst_gpu{ length };

    // random initialization
    src1_cpu.RandomInit(min_val, max_val);
    src2_cpu.RandomInit(min_val, max_val);

    // copy to gpu
    src1_gpu.FromCPU(src1_cpu.data, length);
    src2_gpu.FromCPU(src2_cpu.data, length);

    // run cpu code
    timer_cpu.Start();
    src1_cpu.Add(src2_cpu, dst_cpu);
    timer_cpu.End();

    // run gpu code
    timer_gpu.Start();
    src1_gpu.Add(src2_gpu, dst_gpu);
    timer_gpu.End();

    // print time
    test::PrintTime(timer_cpu, timer_gpu);

    // compare the results
    bool res = test::TestCompare(dst_cpu, dst_gpu, threshold);

    EXPECT_EQ(res, true);
}


// Instantiate the test case with combinations of values
INSTANTIATE_TEST_CASE_P(
    Test_VectorAdd, VectorAdd,
    ::testing::Combine(
        ::testing::Values(7, 100, 5000, 20000, 200000, 2000000),
        ::testing::Values(0.0f, -100.0f),
        ::testing::Values(1.0f, 100.0f)
    ));