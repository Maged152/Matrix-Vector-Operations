cmake_minimum_required(VERSION 3.18)
project(MVOp VERSION 1.0.0 LANGUAGES CUDA CXX)

List (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(PARENT_PROJECT_NAME ${PROJECT_NAME})
add_subdirectory(lib)

# Enable CTest and options only for the main project
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    include(CTest)
    option(MVOp_BUILD_TESTING "Enable testing" ON)
    option(MVOp_BUILD_EXAMPLES "Enable examples" ON)
endif()

# Check if testing is enabled
if(MVOp_BUILD_TESTING AND BUILD_TESTING)
    message(STATUS "Testing for ${PROJECT_NAME} is enabled")
    add_subdirectory(tests)
endif()

# Check if examples are enabled
if(MVOp_BUILD_EXAMPLES)
    message(STATUS "Examples for ${PROJECT_NAME} are enabled")
    add_subdirectory(examples)
endif()
