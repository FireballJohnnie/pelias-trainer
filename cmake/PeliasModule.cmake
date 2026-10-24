# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# pelias_module(<name> [CC <files>] [CU <files>] [DEPS <modules>])
# One static library per module. A module with no source to build (for example kern in a host-only build) becomes an
# INTERFACE library, so the dependency graph is the same with and without CUDA.
include_guard(GLOBAL)
include(CMakeParseArguments)

function(pelias_module NAME)
    cmake_parse_arguments(M "" "" "CC;CU;DEPS" ${ARGN})
    set(srcs ${M_CC})
    if(PELIAS_CUDA)
        list(APPEND srcs ${M_CU})
    endif()
    if(srcs)
        add_library(pelias_${NAME} STATIC ${srcs})
        set(scope PUBLIC)
    else()
        add_library(pelias_${NAME} INTERFACE)
        set(scope INTERFACE)
    endif()
    add_library(pelias::${NAME} ALIAS pelias_${NAME})
    target_include_directories(pelias_${NAME} ${scope} "${PELIAS_SRC_DIR}" "${PELIAS_INCLUDE_DIR}" "${PELIAS_GENERATED_DIR}")
    target_compile_features(pelias_${NAME} ${scope} cxx_std_20)
    foreach(dep IN LISTS M_DEPS)
        target_link_libraries(pelias_${NAME} ${scope} pelias_${dep})
    endforeach()
endfunction()
