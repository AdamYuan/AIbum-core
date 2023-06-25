cmake_minimum_required(VERSION 3.10)
project(aibum)

option(AIBUM_TEST "build tests" ON)
option(AIBUM_PYTHON "build Python modules" ON)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_library(aibum STATIC src/MTCNN.cpp src/FaceNet.cpp src/ImageNet.cpp)
target_include_directories(aibum PUBLIC include)
target_link_libraries(aibum PUBLIC ncnn ${OpenCV_LIBS})

if (AIBUM_TEST)
    add_executable(aibum_class_test test/class.cpp)
    target_link_libraries(aibum_class_test PRIVATE aibum)

    add_executable(aibum_face_test test/face.cpp)
    target_link_libraries(aibum_face_test PRIVATE aibum)
endif ()

if (AIBUM_PYTHON)
    find_package(pybind11 REQUIRED)
endif ()