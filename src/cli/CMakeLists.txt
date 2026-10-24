# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# cli: Command-line entry point.
# pelias-deps: train export verify flow serve
add_executable(pelias_trainer main.cc)
set_target_properties(pelias_trainer PROPERTIES
    OUTPUT_NAME "Pelias-Trainer-v${PELIAS_VERSION_FULL}-${PELIAS_BACKEND}")
target_include_directories(pelias_trainer PRIVATE ${PELIAS_INCLUDE_DIR} ${PELIAS_GENERATED_DIR} ${PELIAS_SRC_DIR})
target_compile_features(pelias_trainer PRIVATE cxx_std_20)
foreach(dep IN ITEMS train export verify flow serve)
    target_link_libraries(pelias_trainer PRIVATE pelias_${dep})
endforeach()
