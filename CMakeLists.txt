cmake_minimum_required(VERSION 3.5)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
    cmake_policy(SET CMP0074 NEW)
endif()

project("jsp_qmt" VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
endif()

find_package(GSL REQUIRED)
find_package(pybind11 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-python REQUIRED)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

file(GLOB_RECURSE header_files "jsp_qmt/*.h", "jsp_qmt/*.txx")
list(SORT header_files)

file(GLOB_RECURSE source_files "jsp_qmt/*.cpp")
list(SORT source_files)

pybind11_add_module(_jsp_qmt SHARED ${source_files} ${header_files})
target_include_directories(
    _jsp_qmt PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
target_link_libraries(
    _jsp_qmt PUBLIC GSL::gsl xtensor xtensor-blas xtensor-python)

file(GLOB_RECURSE python_files "jsp_qmt/*.py")
list(SORT python_files)
add_custom_target(
    PythonFiles ${CMAKE_COMMAND} -E echo "Python files" SOURCES ${python_files})

execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
      -c "import os; \
        from distutils.sysconfig import *; \
        print(get_python_lib(True, prefix='').replace(os.path.sep, '/'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

install(DIRECTORY DESTINATION "${PYTHON_SITE_PACKAGES}")
install(
    TARGETS _jsp_qmt
    DESTINATION "${PYTHON_SITE_PACKAGES}/${CMAKE_PROJECT_NAME}")
install(
    DIRECTORY jsp_qmt
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}
    FILES_MATCHING PATTERN *.py
)
