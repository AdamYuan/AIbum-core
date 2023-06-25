cmake_minimum_required(VERSION 3.10)
project(aibum)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_library(aibum STATIC src/MTCNN.cpp src/FaceNet.cpp src/ImageNet.cpp)
target_include_directories(aibum PUBLIC include)
target_link_libraries(aibum PUBLIC ncnn ${OpenCV_LIBS})

add_executable(aibum_class_test test/class.cpp)
target_link_libraries(aibum_class_test PRIVATE aibum)

add_executable(aibum_face_test test/face.cpp)
target_link_libraries(aibum_face_test PRIVATE aibum)
