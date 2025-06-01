cmake_minimum_required(VERSION 3.22)

include(FetchDependencies)

# Find all test source files
file(GLOB_RECURSE TEST_SOURCES source/*.cpp)

# Create a target for each test file
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} ${PARENT_PROJECT_NAME} GTest::gtest_main ${PARENT_PROJECT_NAME} reference qlm::ThreadPool)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    gtest_discover_tests(${TEST_NAME})

    add_custom_target(Test_${TEST_NAME}
        COMMAND ${CMAKE_COMMAND} -E env
                ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        # DEPENDS ${TEST_NAME}
        COMMENT "Running ${TEST_NAME} test"
    )
endforeach()