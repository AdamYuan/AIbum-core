cmake_minimum_required(VERSION 3.10)
project(aibum-core)

set(CMAKE_CXX_STANDARD 17)

option(AIBUM_TEST "build tests" ON)
option(AIBUM_PYTHON "build Python modules" ON)

find_package(OpenCV REQUIRED)

find_package(ncnn)
if (NOT ncnn_FOUND)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_PYTHON "" OFF)
    option(NCNN_INSTALL_SDK "" OFF)
    add_subdirectory(dep/ncnn)
endif ()

add_library(aibum_core STATIC src/MTCNN.cpp src/FaceNet.cpp src/ImageNet.cpp)
target_include_directories(aibum_core PUBLIC include)
target_link_libraries(aibum_core PUBLIC ncnn ${OpenCV_LIBS})

if (AIBUM_TEST)
    add_executable(aibum_class_test test/class.cpp)
    target_link_libraries(aibum_class_test PRIVATE aibum_core)

    add_executable(aibum_face_test test/face.cpp)
    target_link_libraries(aibum_face_test PRIVATE aibum_core)

    add_executable(aibum_face_cmp_test test/face_cmp.cpp)
    target_link_libraries(aibum_face_cmp_test PRIVATE aibum_core)
endif ()

if (AIBUM_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(pyaibum_core
            python/PyAibum.cpp
            )
    target_link_libraries(pyaibum_core PRIVATE aibum_core)

    install(TARGETS pyaibum_core LIBRARY DESTINATION lib)
endif ()