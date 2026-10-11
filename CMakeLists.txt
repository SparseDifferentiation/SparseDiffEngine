cmake_minimum_required(VERSION 3.15)
project(DNLP_Diff_Engine C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DIFF_ENGINE_VERSION_MAJOR 0)
set(DIFF_ENGINE_VERSION_MINOR 8)
set(DIFF_ENGINE_VERSION_PATCH 0)
set(DIFF_ENGINE_VERSION "${DIFF_ENGINE_VERSION_MAJOR}.${DIFF_ENGINE_VERSION_MINOR}.${DIFF_ENGINE_VERSION_PATCH}")

message(STATUS "Configuring DNLP Differentiation Engine (version ${DIFF_ENGINE_VERSION})")


#Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# MSVC warnings-as-errors: on for standalone builds, off for Python builds so
# downstream toolchains (e.g. conda-forge) aren't broken by new MSVC warnings
if(SKBUILD)
    set(_sde_werror_default OFF)
else()
    set(_sde_werror_default ON)
endif()
option(SDE_WARNINGS_AS_ERRORS "Treat MSVC warnings as errors (/WX)"
       ${_sde_werror_default})

# Sparse linear-algebra library: a self-contained subproject with its own
# CMakeLists.txt, layering checks, BLAS discovery and test suite. Its options are
# set here with plain set() (CMP0077) before the directory is added. Testing is
# enabled first so the root ctest run picks up the library's test. Everything
# directory-scoped below this point (warnings, definitions) is engine-only: it
# reaches only subdirectories added after it.
if(SKBUILD)
    set(SPARSEDIFF_LINALG_BUILD_TESTS OFF)
else()
    enable_testing()
endif()
set(SPARSEDIFF_LINALG_WARNINGS_AS_ERRORS ${SDE_WARNINGS_AS_ERRORS})
# The library is a git submodule (https://github.com/SparseDifferentiation/sparsediff_linalg),
# so a plain clone leaves the directory empty.
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/sparsediff_linalg/CMakeLists.txt")
    message(FATAL_ERROR
        "sparsediff_linalg/ is empty: the sparsediff_linalg submodule is not checked "
        "out. Run\n    git submodule update --init --recursive\n"
        "or clone with --recurse-submodules.")
endif()
add_subdirectory(sparsediff_linalg)

add_compile_definitions(DIFF_ENGINE_VERSION="${DIFF_ENGINE_VERSION}")

# Warning flags (compiler-specific)
if(MSVC)
    add_compile_options(/W4)
    if(SDE_WARNINGS_AS_ERRORS)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(
        -Wall                  # Enable most warnings
        -Wextra                # Extra warnings
        -Wpedantic             # Strict ISO C compliance
        -Wshadow               # Warn about variable shadowing
        -Wformat=2             # Extra format string checks
        -Wcast-qual            # Warn about cast that removes qualifiers
        -Wcast-align           # Warn about pointer cast alignment issues
        -Wunused               # Warn about unused variables/functions
        -Wdouble-promotion     # Warn about float to double promotion
        -Wnull-dereference     # Warn about null pointer dereference
    )
endif()

# Per-config optimization/debug flags, applied identically to every library target
function(sde_set_config_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:/Od /Zi>
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:/O2 /Zi /DNDEBUG>
            $<$<CONFIG:MinSizeRel>:/Os /DNDEBUG>
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-g -O0>
            $<$<CONFIG:Release>:-O3 -DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
            $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
        )
    endif()
endfunction()

# Source files - automatically gather all .c files from src/ (the library lives
# under sparsediff_linalg/, outside this tree).
file(GLOB_RECURSE ENGINE_SOURCES "src/*.c")

# Differentiation engine (expression DAG, atoms, problem). Built
# position-independent so it can be linked into the Python extension module.
add_library(dnlp_diff ${ENGINE_SOURCES})
target_include_directories(dnlp_diff PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dnlp_diff PUBLIC sparsediff_linalg::sparsediff_linalg)
set_property(TARGET dnlp_diff PROPERTY POSITION_INDEPENDENT_CODE ON)
sde_set_config_options(dnlp_diff)

# This is needed for clock_gettime on Linux without compiler extensions
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_definitions(_POSIX_C_SOURCE=200809L)
endif()

# =============================================================================
# C tests (only for standalone builds)
# =============================================================================
option(PROFILE_ONLY "Build only profiling tests" OFF)

if(NOT SKBUILD)
    add_executable(all_tests
        tests/all_tests.c
        tests/test_helpers.c
        tests/numerical_diff.c
    )
    target_include_directories(all_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    target_link_libraries(all_tests dnlp_diff)
    
    if(PROFILE_ONLY)
        target_compile_definitions(all_tests PRIVATE PROFILE_ONLY)
        message(STATUS "Building ONLY profiling tests")
    endif()
    
    add_test(NAME AllTests COMMAND all_tests)
endif()
