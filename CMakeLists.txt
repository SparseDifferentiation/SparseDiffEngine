cmake_minimum_required(VERSION 3.10)
project(DNLP_Diff_Engine C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_BUILD_TYPE Debug)

# Debug-friendly flags
add_compile_options(-g -O0)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/tests)

# Source files - automatically gather all .c files from src/
file(GLOB_RECURSE SOURCES "src/*.c")

# Create library
add_library(dnlp_diff ${SOURCES})
target_link_libraries(dnlp_diff m)

# Enable testing
enable_testing()

# Single test executable combining all tests
add_executable(all_tests
    tests/all_tests.c
    tests/test_helpers.c
)
target_link_libraries(all_tests dnlp_diff)
add_test(NAME AllTests COMMAND all_tests)
