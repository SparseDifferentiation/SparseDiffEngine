cmake_minimum_required(VERSION 3.15)
project(DNLP_Diff_Engine C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DIFF_ENGINE_VERSION_MAJOR 0)
set(DIFF_ENGINE_VERSION_MINOR 8)
set(DIFF_ENGINE_VERSION_PATCH 0)
set(DIFF_ENGINE_VERSION "${DIFF_ENGINE_VERSION_MAJOR}.${DIFF_ENGINE_VERSION_MINOR}.${DIFF_ENGINE_VERSION_PATCH}")
add_compile_definitions(DIFF_ENGINE_VERSION="${DIFF_ENGINE_VERSION}")

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

# Source files - automatically gather all .c files from src/. The sparse
# linear-algebra layer (src/sparse_linalg, include/sparse_linalg) is built as its
# own library; everything else under src/ is the differentiation engine.
file(GLOB_RECURSE SPARSE_LINALG_SOURCES "src/sparse_linalg/*.c")
file(GLOB_RECURSE SPARSE_LINALG_HEADERS "include/sparse_linalg/*.h" "src/sparse_linalg/*.h")
file(GLOB_RECURSE ENGINE_SOURCES "src/*.c")
list(FILTER ENGINE_SOURCES EXCLUDE REGEX "/src/sparse_linalg/")

# Layering rule: sparse_linalg is a standalone sparse linear-algebra layer and
# must not depend on the differentiation engine. Every quoted include in a
# library file must name a library header: a public one
# ("sparse_linalg/<name>.h", under include/) or a private one
# ("sparse_linalg/internal/<name>.h", under src/sparse_linalg/internal/, which
# only the library and its own test suite can resolve). A public header must
# never include a private one, or "public" would stop meaning anything. System
# headers use <...>. Checked here so a violation fails every configure, locally
# and in CI.
foreach(_sde_file IN LISTS SPARSE_LINALG_SOURCES SPARSE_LINALG_HEADERS)
    file(STRINGS "${_sde_file}" _sde_includes REGEX "^[ \t]*#[ \t]*include[ \t]*\"")
    foreach(_sde_line IN LISTS _sde_includes)
        if(NOT _sde_line MATCHES "\"sparse_linalg/(internal/)?[A-Za-z0-9_]+\\.h\"")
            message(FATAL_ERROR
                "Layering violation in ${_sde_file}:\n"
                "    ${_sde_line}\n"
                "sparse_linalg may include only its own headers "
                "(#include \"sparse_linalg/<name>.h\" or "
                "\"sparse_linalg/internal/<name>.h\") and system headers.")
        endif()
        if(_sde_file MATCHES "/include/sparse_linalg/"
           AND _sde_line MATCHES "\"sparse_linalg/internal/")
            message(FATAL_ERROR
                "Public header ${_sde_file} includes a private one:\n"
                "    ${_sde_line}\n"
                "Headers under include/sparse_linalg must not include "
                "sparse_linalg/internal/ headers.")
        endif()
    endforeach()
endforeach()

# The library test suite, its shared helpers and the bench build against
# sparse_linalg alone. include/ is public on the target, so the linker would
# not catch a header-only reach into the engine; this does. Quoted includes may
# name a library header, minunit/test_helpers, or a sibling test_<name>.h.
file(GLOB_RECURSE SPARSE_LINALG_TEST_FILES
    "tests/sparse_linalg/*.c" "tests/sparse_linalg/*.h" "bench/sparse_linalg/*.c")
list(APPEND SPARSE_LINALG_TEST_FILES
    "${PROJECT_SOURCE_DIR}/tests/minunit.h"
    "${PROJECT_SOURCE_DIR}/tests/test_helpers.h"
    "${PROJECT_SOURCE_DIR}/tests/test_helpers.c")
foreach(_sde_file IN LISTS SPARSE_LINALG_TEST_FILES)
    file(STRINGS "${_sde_file}" _sde_includes REGEX "^[ \t]*#[ \t]*include[ \t]*\"")
    foreach(_sde_line IN LISTS _sde_includes)
        if(NOT _sde_line MATCHES
           "\"(sparse_linalg/(internal/)?[A-Za-z0-9_]+|minunit|test_[A-Za-z0-9_]+)\\.h\"")
            message(FATAL_ERROR
                "Layering violation in ${_sde_file}:\n"
                "    ${_sde_line}\n"
                "sparse_linalg tests may include only sparse_linalg/<name>.h, "
                "minunit.h, test_helpers.h and sibling test_<name>.h files.")
        endif()
    endforeach()
endforeach()

# Both libraries are built position-independent so they can be linked into the
# Python extension module.

# Sparse linear-algebra library
add_library(sparse_linalg ${SPARSE_LINALG_SOURCES})
target_include_directories(sparse_linalg PUBLIC ${PROJECT_SOURCE_DIR}/include)
# src/ resolves the private headers ("sparse_linalg/internal/<name>.h"); it is
# PRIVATE, so consumers such as dnlp_diff cannot include them at all.
target_include_directories(sparse_linalg PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET sparse_linalg PROPERTY POSITION_INDEPENDENT_CODE ON)
sde_set_config_options(sparse_linalg)

# Differentiation engine (expression DAG, atoms, problem)
add_library(dnlp_diff ${ENGINE_SOURCES})
target_link_libraries(dnlp_diff PUBLIC sparse_linalg)
set_property(TARGET dnlp_diff PROPERTY POSITION_INDEPENDENT_CODE ON)
sde_set_config_options(dnlp_diff)

# Optional peak-memory tracking via the sp_malloc/sp_free counters. OFF by default
# so the library carries no mutable global state. PUBLIC on sparse_linalg so every
# consumer of tracked_alloc.h (engine, tests, Python extension) sees the same
# definition.
option(SP_TRACK_MEMORY "Track peak memory usage via sp_malloc/sp_free counters" OFF)
if(SP_TRACK_MEMORY)
    target_compile_definitions(sparse_linalg PUBLIC SP_TRACK_MEMORY)
endif()

# Link math library (Unix/Linux only)
if(NOT MSVC)
    target_link_libraries(sparse_linalg PUBLIC m)
endif()

# Find and link BLAS. Linked PUBLIC on sparse_linalg: the library's own kernels
# call CBLAS, and so do a few engine atoms, which inherit it through dnlp_diff.
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    target_link_libraries(sparse_linalg PUBLIC ${ACCELERATE_FRAMEWORK})
elseif(MSVC)
    find_package(OpenBLAS CONFIG QUIET)
    if(TARGET OpenBLAS::OpenBLAS)
        # vcpkg
        target_link_libraries(sparse_linalg PUBLIC OpenBLAS::OpenBLAS)
    else()
        # e.g. conda-forge's openblas ships openblas.lib and
        # include/openblas/cblas.h, but no imported target
        find_library(OPENBLAS_LIBRARY NAMES openblas libopenblas REQUIRED)
        get_filename_component(_sde_openblas_libdir "${OPENBLAS_LIBRARY}" DIRECTORY)
        get_filename_component(_sde_openblas_prefix "${_sde_openblas_libdir}" DIRECTORY)
        find_path(OPENBLAS_INCLUDE_DIR cblas.h
                  HINTS "${_sde_openblas_prefix}/include"
                  PATH_SUFFIXES openblas REQUIRED)
        target_link_libraries(sparse_linalg PUBLIC ${OPENBLAS_LIBRARY})
        target_include_directories(sparse_linalg PUBLIC ${OPENBLAS_INCLUDE_DIR})
    endif()
else()
    find_package(BLAS REQUIRED)
    # conda-forge splits the CBLAS interface into libcblas next to libblas. Link
    # it explicitly (and before BLAS) so the package keeps working when users
    # switch BLAS implementations, even if the build-time libblas exports cblas_*.
    set(_sde_cblas_include_hints)
    if(BLAS_LIBRARIES)
        list(GET BLAS_LIBRARIES 0 _sde_blas_lib)
        get_filename_component(_sde_blas_dir "${_sde_blas_lib}" DIRECTORY)
        find_library(CBLAS_LIBRARY NAMES cblas
                     HINTS "${_sde_blas_dir}" NO_DEFAULT_PATH)
        list(APPEND _sde_cblas_include_hints "${_sde_blas_dir}/../include")
    endif()
    if(CBLAS_LIBRARY)
        target_link_libraries(sparse_linalg PUBLIC ${CBLAS_LIBRARY})
    endif()
    target_link_libraries(sparse_linalg PUBLIC ${BLAS_LIBRARIES})
    # Some distributions ship cblas.h only under include/openblas. Export the
    # directory when it is found so library and engine sources both compile; a
    # miss is not an error because cblas.h is normally on the default path.
    find_path(CBLAS_INCLUDE_DIR cblas.h
              HINTS ${_sde_cblas_include_hints}
              PATH_SUFFIXES openblas)
    if(CBLAS_INCLUDE_DIR)
        target_include_directories(sparse_linalg PUBLIC ${CBLAS_INCLUDE_DIR})
    endif()
endif()

# This is needed for clock_gettime on Linux without compiler extensions
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_definitions(_POSIX_C_SOURCE=200809L)
endif()

# =============================================================================
# C tests (only for standalone builds)
# =============================================================================
option(PROFILE_ONLY "Build only profiling tests" OFF)

if(NOT SKBUILD)
    enable_testing()

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

    # Library suite: links sparse_linalg alone, so a test that reaches into the
    # engine or old-code fails to link. Not gated by PROFILE_ONLY: it has no
    # profiling content and a profile-only build still validates the library
    # it profiles. The sibling test headers resolve relative to
    # sparse_linalg_tests.c; src/ is added because the library's own suite is
    # white-box and tests the private kernels too.
    add_executable(sparse_linalg_tests
        tests/sparse_linalg/sparse_linalg_tests.c
        tests/test_helpers.c
    )
    target_include_directories(sparse_linalg_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/src
    )
    target_link_libraries(sparse_linalg_tests sparse_linalg)

    add_test(NAME SparseLinalgTests COMMAND sparse_linalg_tests)
endif()
