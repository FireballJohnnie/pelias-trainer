# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# Options, language standard, warnings and the CUDA switch.
option(PELIAS_CUDA "Build the CUDA parts (needs nvcc)" ON)
option(PELIAS_BUILD_TESTS "Build the tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra)
endif()

if(PELIAS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "PELIAS_CUDA is ON but no CUDA compiler was found. "
                            "Install the CUDA toolkit, or configure with -DPELIAS_CUDA=OFF (preset host-only) for a build without GPU code.")
    endif()
    # sm_89 only (Ada); tile constants come from the probes (master 5.1).
    set(CMAKE_CUDA_ARCHITECTURES 89)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")
    enable_language(CUDA)
    set(PELIAS_BACKEND "cuda")
else()
    set(PELIAS_BACKEND "host")
endif()
