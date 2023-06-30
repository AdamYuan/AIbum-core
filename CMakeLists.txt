cmake_minimum_required(VERSION 3.10)
project(aibum-core)

set(CMAKE_CXX_STANDARD 17)

option(AIBUM_WASM_ONLY "build WASM only" ON)
option(AIBUM_TEST "build tests" ON)

add_library(aibum_core STATIC
        src/SCRFD.cpp src/FaceNet.cpp src/ImageNet.cpp
        models/src/mobilefacenet.mem.c
        models/src/mobilenet_v3.mem.c
        models/src/scrfd.mem.c
        )
target_include_directories(aibum_core PUBLIC include PRIVATE models/include)

if (NOT AIBUM_WASM_ONLY)
    find_package(ncnn REQUIRED)
    target_link_libraries(aibum_core PUBLIC ncnn)

    if (AIBUM_TEST)
        find_package(OpenCV REQUIRED)

        add_executable(aibum_class_test test/class.cpp)
        target_link_libraries(aibum_class_test PRIVATE aibum_core ${OpenCV_LIBS})

        add_executable(aibum_face_test test/face.cpp)
        target_link_libraries(aibum_face_test PRIVATE aibum_core ${OpenCV_LIBS})

        add_executable(aibum_face_cmp_test test/face_cmp.cpp)
        target_link_libraries(aibum_face_cmp_test PRIVATE aibum_core ${OpenCV_LIBS})
    endif ()
else ()
    add_library(ncnn INTERFACE)
    target_link_libraries(ncnn INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/wasm/dep/ncnn/simd-threads/lib/libncnn.a")
    target_include_directories(ncnn INTERFACE wasm/dep/ncnn/simd-threads/include wasm/dep/stb)

    target_link_libraries(aibum_core PUBLIC ncnn)

    add_executable(aibum_core_wasm
            wasm/aibum_core_wasm.cpp
            )
    target_link_libraries(aibum_core_wasm PRIVATE aibum_core)
    target_link_options(aibum_core_wasm PRIVATE
            --bind
            -sENVIRONMENT=web
            -sMODULARIZE=1
            -sEXPORT_ES6=1
            -sDISABLE_EXCEPTION_CATCHING=1
            -sEXPORT_NAME=loadAIbumCore
            -sALLOW_MEMORY_GROWTH
            -sEXPORTED_FUNCTIONS=[_malloc]
            -sTOTAL_MEMORY=32MB
            )

    install(FILES
            "$<TARGET_FILE_DIR:aibum_core_wasm>/aibum_core_wasm.js"
            "$<TARGET_FILE_DIR:aibum_core_wasm>/aibum_core_wasm.wasm"
            )
endif ()
