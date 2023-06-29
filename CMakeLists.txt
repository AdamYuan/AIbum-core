cmake_minimum_required(VERSION 3.10)
project(aibum-core)

set(CMAKE_CXX_STANDARD 17)

option(AIBUM_WASM_ONLY "build WASM only" ON)
option(AIBUM_TEST "build tests" ON)
option(AIBUM_PYTHON "build Python modules" ON)

if (NOT AIBUM_WASM_ONLY)
    find_package(OpenCV REQUIRED)

    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_PYTHON "" OFF)
    option(NCNN_INSTALL_SDK "" OFF)
    add_subdirectory(dep/ncnn)

    add_library(aibum_core STATIC
            src/SCRFD.cpp src/FaceNet.cpp src/ImageNet.cpp
            models/src/mobilefacenet.mem.c
            models/src/mobilenet_v3.mem.c
            models/src/scrfd.mem.c
            )
    target_include_directories(aibum_core PUBLIC include PRIVATE models/include)
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
        find_package(pybind11 CONFIG)
        if (NOT pybind11_FOUND)
            add_subdirectory(dep/pybind11)
        endif ()

        pybind11_add_module(pyaibum_core
                python/PyAibum.cpp
                )
        target_link_libraries(pyaibum_core PRIVATE aibum_core)
    endif ()
else ()
    add_executable(aibum_core_wasm src/SCRFD.cpp src/FaceNet.cpp src/ImageNet.cpp)
	target_include_directories(aibum_core_wasm PRIVATE include models/include wasm/dep/ncnn/include)
    target_link_libraries(aibum_core_wasm PRIVATE wasm/dep/ncnn/lib/libncnn.a)
endif ()
