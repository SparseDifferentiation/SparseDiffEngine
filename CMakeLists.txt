cmake_minimum_required(VERSION 3.15)
project(DNLP_Diff_Engine C)

set(CMAKE_C_STANDARD 99)

# Set build type if not specified (standalone builds)
if(NOT CMAKE_BUILD_TYPE AND NOT SKBUILD)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Warning flags (always enabled)
add_compile_options(
    -Wall                  # Enable most warnings
    -Wextra                # Extra warnings
    -Wpedantic             # Strict ISO C compliance
    -Wshadow               # Warn about variable shadowing
    -Wformat=2             # Extra format string checks
    -Wcast-qual            # Warn about cast that removes qualifiers
    -Wcast-align           # Warn about pointer cast alignment issues
    -Wunused               # Warn about unused variables/functions
    -Wdouble-promotion     # Warn about float->double promotion
    -Wnull-dereference     # Warn about null pointer dereference
)

# Debug flags only for standalone debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT SKBUILD)
    add_compile_options(-g -O0)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files - automatically gather all .c files from src/
file(GLOB_RECURSE SOURCES "src/*.c")

# Create core library
add_library(dnlp_diff ${SOURCES})
target_link_libraries(dnlp_diff m)

# =============================================================================
# Python bindings (built when using scikit-build-core)
# =============================================================================
if(SKBUILD)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

    # Create Python extension module
    Python3_add_library(_core MODULE python/bindings.c)
    target_include_directories(_core PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/python
        ${Python3_NumPy_INCLUDE_DIRS}
    )
    target_link_libraries(_core PRIVATE dnlp_diff)

    # Install to the package directory
    install(TARGETS _core LIBRARY DESTINATION dnlp_diff_engine)
endif()

# =============================================================================
# C tests (only for standalone builds)
# =============================================================================
if(NOT SKBUILD)
    include_directories(${PROJECT_SOURCE_DIR}/tests)
    enable_testing()

    add_executable(all_tests
        tests/all_tests.c
        tests/test_helpers.c
    )
    target_link_libraries(all_tests dnlp_diff)
    add_test(NAME AllTests COMMAND all_tests)
endif()
