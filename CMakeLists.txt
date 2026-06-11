cmake_minimum_required(VERSION 3.18)
project(py_ross LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ROSS REQUIRED)
find_package(MPI REQUIRED COMPONENTS C)

# nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT NB_DIR)
  execute_process(
    COMMAND python -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_ross
  STABLE_ABI
  NB_STATIC
  src/module.cpp
)

target_link_libraries(_ross PRIVATE ROSS MPI::MPI_C)

install(TARGETS _ross LIBRARY DESTINATION ross)
