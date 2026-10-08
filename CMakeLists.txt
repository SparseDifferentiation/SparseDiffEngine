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

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files - automatically gather all .c files from src/
file(GLOB_RECURSE SOURCES "src/*.c")


# Create core library
add_library(dnlp_diff ${SOURCES})

# Optional peak-memory tracking via the sp_malloc/sp_free counters. OFF by default
# so the library carries no mutable global state.
option(SP_TRACK_MEMORY "Track peak memory usage via sp_malloc/sp_free counters" OFF)
if(SP_TRACK_MEMORY)
    target_compile_definitions(dnlp_diff PUBLIC SP_TRACK_MEMORY)
endif()

# Link math library (Unix/Linux only)
if(NOT MSVC)
    target_link_libraries(dnlp_diff m)
endif()

# Find and link BLAS
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    target_link_libraries(dnlp_diff ${ACCELERATE_FRAMEWORK})
elseif(MSVC)
    find_package(OpenBLAS CONFIG QUIET)
    if(TARGET OpenBLAS::OpenBLAS)
        # vcpkg
        target_link_libraries(dnlp_diff OpenBLAS::OpenBLAS)
    else()
        # e.g. conda-forge's openblas ships openblas.lib and
        # include/openblas/cblas.h, but no imported target
        find_library(OPENBLAS_LIBRARY NAMES openblas libopenblas REQUIRED)
        get_filename_component(_sde_openblas_libdir "${OPENBLAS_LIBRARY}" DIRECTORY)
        get_filename_component(_sde_openblas_prefix "${_sde_openblas_libdir}" DIRECTORY)
        find_path(OPENBLAS_INCLUDE_DIR cblas.h
                  HINTS "${_sde_openblas_prefix}/include"
                  PATH_SUFFIXES openblas REQUIRED)
        target_link_libraries(dnlp_diff ${OPENBLAS_LIBRARY})
        target_include_directories(dnlp_diff PUBLIC ${OPENBLAS_INCLUDE_DIR})
    endif()
else()
    find_package(BLAS REQUIRED)
    # conda-forge splits the CBLAS interface into libcblas next to libblas. Link
    # it explicitly (and before BLAS) so the package keeps working when users
    # switch BLAS implementations, even if the build-time libblas exports cblas_*.
    if(BLAS_LIBRARIES)
        list(GET BLAS_LIBRARIES 0 _sde_blas_lib)
        get_filename_component(_sde_blas_dir "${_sde_blas_lib}" DIRECTORY)
        find_library(CBLAS_LIBRARY NAMES cblas
                     HINTS "${_sde_blas_dir}" NO_DEFAULT_PATH)
    endif()
    if(CBLAS_LIBRARY)
        target_link_libraries(dnlp_diff ${CBLAS_LIBRARY})
    endif()
    target_link_libraries(dnlp_diff ${BLAS_LIBRARIES})
endif()

# Config-specific compile options (compiler-specific)
if(MSVC)
    target_compile_options(dnlp_diff PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi /DNDEBUG>
        $<$<CONFIG:MinSizeRel>:/Os /DNDEBUG>
    )
else()
    target_compile_options(dnlp_diff PRIVATE
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
        $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    )
endif()

# This is needed for clock_gettime on Linux without compiler extensions
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_definitions(_POSIX_C_SOURCE=200809L)
endif()

# Enable position-independent code for shared library compatibility
set_property(TARGET dnlp_diff PROPERTY POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# C tests (only for standalone builds)
# =============================================================================
option(PROFILE_ONLY "Build only profiling tests" OFF)

if(NOT SKBUILD)
    include_directories(${PROJECT_SOURCE_DIR}/tests)
    enable_testing()

    add_executable(all_tests
        tests/all_tests.c
        tests/test_helpers.c
        tests/numerical_diff.c
    )
    target_link_libraries(all_tests dnlp_diff)
    
    if(PROFILE_ONLY)
        target_compile_definitions(all_tests PRIVATE PROFILE_ONLY)
        message(STATUS "Building ONLY profiling tests")
    endif()
    
    add_test(NAME AllTests COMMAND all_tests)
endif()
