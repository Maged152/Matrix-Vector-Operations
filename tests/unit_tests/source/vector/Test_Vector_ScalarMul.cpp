#include "test_helper.hpp"
#include "matrix_vector_op.hpp"
#include "reference.hpp"


// Define the test parameters types
struct VectorScalarMul : ::testing::TestWithParam<std::tuple<
    int,   // length
    float, // min value
    float  // max value
    >>
{};


// Define a parameterized test case
TEST_P(VectorScalarMul, Test_VectorScalarMul)
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

    // input vector
    qlm::Vector<qlm::MemType::MEM_UM> src{ length };

    // scalar value
    qlm::Array<1, qlm::MemType::MEM_UM> val;
    val.Set(0, 5.0f);

    // output vectors
    qlm::Vector<qlm::MemType::MEM_UM> dst_cpu{ length };
    qlm::Vector<qlm::MemType::MEM_UM> dst_gpu{ length };

    // random initialization
    src.RandomInit(min_val, max_val);

    // run cpu code
    timer_cpu.Start();
    test::Mul(src, val.Get(0), dst_cpu);
    timer_cpu.End();

    // run gpu code
    timer_gpu.Start();
    qlm::Mul(src, val, dst_gpu);
    timer_gpu.End();

    // print time
    test::PrintTime(timer_cpu, timer_gpu);

    // compare the results
    bool res = test::TestCompare(dst_cpu, dst_gpu, threshold);

    EXPECT_EQ(res, true);
}


// Instantiate the test case with combinations of values
INSTANTIATE_TEST_CASE_P(
    Test_VectorScalarMul, VectorScalarMul,
    ::testing::Combine(
        ::testing::Values(7, 100, 5000, 20000, 200000, 2000000),
        ::testing::Values(0.0f, -100.0f),
        ::testing::Values(1.0f, 100.0f)
    ));