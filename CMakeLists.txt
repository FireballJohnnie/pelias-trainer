# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
cmake_minimum_required(VERSION 3.28)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PELIAS_VERSION_FULL LIMIT_COUNT 1)
project(pelias_trainer VERSION ${PELIAS_VERSION_FULL} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PeliasOptions)
include(PeliasModule)

set(PELIAS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PELIAS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PELIAS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file("${PELIAS_INCLUDE_DIR}/pelias/version.h.in" "${PELIAS_GENERATED_DIR}/pelias/version.h" @ONLY)

add_subdirectory(src)

if(PELIAS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "Pelias-Trainer-v${PELIAS_VERSION_FULL}-${PELIAS_BACKEND}")
