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

find_package(gsl REQUIRED)
find_package(pybind11 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-python REQUIRED)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
function(build_module module header_files source_files)
    pybind11_add_module(${module} SHARED ${source_files} ${header_files})
    target_include_directories(
        ${module} PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
    target_link_libraries(
        ${module} PUBLIC GSL::gsl xtensor xtensor-blas xtensor-python)
endfunction()

set(header_files jsp_qmt/brentq.h)
set(source_files jsp_qmt/_MTsat.cpp)
build_module(_MTsat "${header_files}" "${source_files}")

set(
    header_files
    jsp_qmt/brentq.h jsp_qmt/expm.h jsp_qmt/SPqMT.h jsp_qmt/SPqMT.txx
    jsp_qmt/super_lorentzian.h)
set(
    source_files
    jsp_qmt/_SPqMT.cpp jsp_qmt/expm.cpp jsp_qmt/SPqMT.cpp
    jsp_qmt/super_lorentzian.cpp)
build_module(_SPqMT "${header_files}" "${source_files}")

file(GLOB_RECURSE python_files "jsp_qmt/*.py")
list(SORT python_files)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
      -c "import os; \
        from distutils.sysconfig import *; \
        print(get_python_lib(True, prefix='').replace(os.path.sep, '/'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

install(DIRECTORY DESTINATION "${PYTHON_SITE_PACKAGES}")
install(
    TARGETS _MTsat _SPqMT
    DESTINATION "${PYTHON_SITE_PACKAGES}/${CMAKE_PROJECT_NAME}")
install(
    FILES ${python_files}
    DESTINATION "${PYTHON_SITE_PACKAGES}/${CMAKE_PROJECT_NAME}")
