#include "test_helper.hpp"
#include "matrix_vector_op.hpp"
#include "reference.hpp"


// Define the test parameters types
struct VectorMean : ::testing::TestWithParam<std::tuple<
    int,   // length
    float, // min value
    float  // max value
    >>
{};


// Define a parameterized test case
TEST_P(VectorMean, Test_VectorMean)
{
    constexpr float threshold = 60.0f;
    // extract the parameters
    auto& [length, min_val, max_val] = GetParam();

    // print the parameters
    test::PrintParameter(length, "length");
    test::PrintParameter(min_val, "min_val");
    test::PrintParameter(max_val, "max_val");

    qlm::Timer<qlm::usec> timer_cpu;
    qlm::Timer<qlm::usec> timer_gpu;

    float dst_cpu;
    qlm::Array<1, qlm::MemType::MEM_UM> dst_gpu;

    // input vector
    qlm::Vector<qlm::MemType::MEM_UM> src{ length };

    // random initialization
    src.RandomInit(min_val, max_val);

    // run cpu code
    timer_cpu.Start();
    test::Mean(src, dst_cpu);
    timer_cpu.End();

    // run gpu code
    timer_gpu.Start();
    qlm::Mean(src, dst_gpu);
    timer_gpu.End();

    // print time
    test::PrintTime(timer_cpu, timer_gpu);

    // compare the results
    bool res = test::TestCompare_SNR(dst_cpu, dst_gpu, threshold);

    EXPECT_EQ(res, true);
}


// Instantiate the test case with combinations of values
INSTANTIATE_TEST_CASE_P(
    Test_VectorMean, VectorMean,
    ::testing::Combine(
        ::testing::Values(7, 256, 5000, 20000, 200000, 2000000),
        ::testing::Values(0.0f, -100.0f),
        ::testing::Values(1.0f, 100.0f)
    ));