cmake_minimum_required(VERSION 3.15)
project(sparse_linalg LANGUAGES C)
set(CMAKE_C_STANDARD 99)

# Standalone (top-level) build vs. embedded via add_subdirectory(): a parent
# project owns the build type and may override the options below with plain
# set() before adding this directory (CMP0077).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SPARSE_LINALG_TOP_LEVEL ON)
else()
    set(SPARSE_LINALG_TOP_LEVEL OFF)
endif()

if(SPARSE_LINALG_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(SPARSE_LINALG_BUILD_TESTS "Build the sparse_linalg test suite" ON)
option(SPARSE_LINALG_WARNINGS_AS_ERRORS "Treat MSVC warnings as errors (/WX)" ON)
# Optional peak-memory tracking via the sp_malloc/sp_free counters. OFF by default
# so the library carries no mutable global state. PUBLIC on the target so every
# consumer of tracked_alloc.h sees the same definition.
option(SP_TRACK_MEMORY "Track peak memory usage via sp_malloc/sp_free counters" OFF)

# Warning flags, applied per target so they neither depend on nor leak into a
# parent project.
function(sparse_linalg_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(SPARSE_LINALG_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE
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
endfunction()

file(GLOB_RECURSE SPARSE_LINALG_SOURCES "src/*.c")
file(GLOB_RECURSE SPARSE_LINALG_HEADERS "include/sparse_linalg/*.h" "src/internal/*.h")

# Layering rule: sparse_linalg is a standalone sparse linear-algebra layer and
# must not depend on whatever embeds it. Every quoted include in a library file
# must name a library header: a public one ("sparse_linalg/<name>.h", under
# include/) or a private one ("internal/<name>.h", under src/internal/, which
# only the library and its own test suite can resolve). A public header must
# never include a private one, or "public" would stop meaning anything. System
# headers use <...>. Checked here so a violation fails every configure, locally
# and in CI.
foreach(_sl_file IN LISTS SPARSE_LINALG_SOURCES SPARSE_LINALG_HEADERS)
    file(STRINGS "${_sl_file}" _sl_includes REGEX "^[ \t]*#[ \t]*include[ \t]*\"")
    foreach(_sl_line IN LISTS _sl_includes)
        if(NOT _sl_line MATCHES "\"(sparse_linalg|internal)/[A-Za-z0-9_]+\\.h\"")
            message(FATAL_ERROR
                "Layering violation in ${_sl_file}:\n"
                "    ${_sl_line}\n"
                "sparse_linalg may include only its own headers "
                "(#include \"sparse_linalg/<name>.h\" or "
                "\"internal/<name>.h\") and system headers.")
        endif()
        if(_sl_file MATCHES "/include/sparse_linalg/" AND _sl_line MATCHES "\"internal/")
            message(FATAL_ERROR
                "Public header ${_sl_file} includes a private one:\n"
                "    ${_sl_line}\n"
                "Headers under include/sparse_linalg must not include "
                "internal/ headers.")
        endif()
    endforeach()
endforeach()

# Sparse linear-algebra library. Built position-independent so it can be linked
# into shared objects such as a Python extension module.
add_library(sparse_linalg ${SPARSE_LINALG_SOURCES})
add_library(sparse_linalg::sparse_linalg ALIAS sparse_linalg)
target_include_directories(sparse_linalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# src/ resolves the private headers ("internal/<name>.h"); it is PRIVATE, so
# consumers cannot include them at all.
target_include_directories(sparse_linalg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_property(TARGET sparse_linalg PROPERTY POSITION_INDEPENDENT_CODE ON)
sparse_linalg_warnings(sparse_linalg)

if(SP_TRACK_MEMORY)
    target_compile_definitions(sparse_linalg PUBLIC SP_TRACK_MEMORY)
endif()

# Link math library (Unix/Linux only)
if(NOT MSVC)
    target_link_libraries(sparse_linalg PUBLIC m)
endif()

# Find and link BLAS. Linked PUBLIC: the library's own kernels call CBLAS, and so
# may consumers, which inherit it through the target.
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
        get_filename_component(_sl_openblas_libdir "${OPENBLAS_LIBRARY}" DIRECTORY)
        get_filename_component(_sl_openblas_prefix "${_sl_openblas_libdir}" DIRECTORY)
        find_path(OPENBLAS_INCLUDE_DIR cblas.h
                  HINTS "${_sl_openblas_prefix}/include"
                  PATH_SUFFIXES openblas REQUIRED)
        target_link_libraries(sparse_linalg PUBLIC ${OPENBLAS_LIBRARY})
        target_include_directories(sparse_linalg PUBLIC ${OPENBLAS_INCLUDE_DIR})
    endif()
else()
    find_package(BLAS REQUIRED)
    # conda-forge splits the CBLAS interface into libcblas next to libblas. Link
    # it explicitly (and before BLAS) so the package keeps working when users
    # switch BLAS implementations, even if the build-time libblas exports cblas_*.
    set(_sl_cblas_include_hints)
    if(BLAS_LIBRARIES)
        list(GET BLAS_LIBRARIES 0 _sl_blas_lib)
        get_filename_component(_sl_blas_dir "${_sl_blas_lib}" DIRECTORY)
        find_library(CBLAS_LIBRARY NAMES cblas
                     HINTS "${_sl_blas_dir}" NO_DEFAULT_PATH)
        list(APPEND _sl_cblas_include_hints "${_sl_blas_dir}/../include")
    endif()
    if(CBLAS_LIBRARY)
        target_link_libraries(sparse_linalg PUBLIC ${CBLAS_LIBRARY})
    endif()
    target_link_libraries(sparse_linalg PUBLIC ${BLAS_LIBRARIES})
    # Some distributions ship cblas.h only under include/openblas. Export the
    # directory when it is found so library and consumer sources both compile; a
    # miss is not an error because cblas.h is normally on the default path.
    find_path(CBLAS_INCLUDE_DIR cblas.h
              HINTS ${_sl_cblas_include_hints}
              PATH_SUFFIXES openblas)
    if(CBLAS_INCLUDE_DIR)
        target_include_directories(sparse_linalg PUBLIC ${CBLAS_INCLUDE_DIR})
    endif()
endif()

# =============================================================================
# Tests
# =============================================================================
if(SPARSE_LINALG_BUILD_TESTS)
    enable_testing()

    # The suite links sparse_linalg alone, so a test that reaches outside the
    # library fails to link. include/ is public on the target, so the linker
    # would not catch a header-only reach; this check does. Quoted includes may
    # name a library header (public or private: the suite is white-box),
    # minunit/test_helpers, or a sibling test_<name>.h.
    file(GLOB_RECURSE _sl_test_files "tests/*.c" "tests/*.h")
    foreach(_sl_file IN LISTS _sl_test_files)
        file(STRINGS "${_sl_file}" _sl_includes REGEX "^[ \t]*#[ \t]*include[ \t]*\"")
        foreach(_sl_line IN LISTS _sl_includes)
            if(NOT _sl_line MATCHES
               "\"((sparse_linalg|internal)/[A-Za-z0-9_]+|minunit|test_[A-Za-z0-9_]+)\\.h\"")
                message(FATAL_ERROR
                    "Layering violation in ${_sl_file}:\n"
                    "    ${_sl_line}\n"
                    "sparse_linalg tests may include only sparse_linalg/<name>.h, "
                    "internal/<name>.h, minunit.h, test_helpers.h and sibling "
                    "test_<name>.h files.")
            endif()
        endforeach()
    endforeach()

    add_executable(sparse_linalg_tests
        tests/sparse_linalg_tests.c
        tests/test_helpers.c
    )
    target_include_directories(sparse_linalg_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(sparse_linalg_tests sparse_linalg)
    sparse_linalg_warnings(sparse_linalg_tests)

    add_test(NAME SparseLinalgTests COMMAND sparse_linalg_tests)
endif()
