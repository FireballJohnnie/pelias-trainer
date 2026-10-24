# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
add_executable(framework_smoke smoke/framework_smoke.cc)
target_include_directories(framework_smoke PRIVATE ${PELIAS_INCLUDE_DIR} ${PELIAS_GENERATED_DIR} ${PELIAS_SRC_DIR})
target_compile_features(framework_smoke PRIVATE cxx_std_20)
foreach(mod IN ITEMS hal tok io plan data model stream kern sched train export verify project startup intake help warn flow serve ui)
    target_link_libraries(framework_smoke PRIVATE pelias_${mod})
endforeach()
add_test(NAME framework_smoke COMMAND framework_smoke)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME module_dependencies COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/check_module_deps.py" "${CMAKE_SOURCE_DIR}")
    add_test(NAME resources COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/check_resources.py" "${CMAKE_SOURCE_DIR}")
else()
    message(STATUS "Python 3 not found: the module_dependencies and resources checks are not registered")
endif()
