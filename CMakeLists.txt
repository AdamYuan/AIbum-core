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
    find_package(pybind11 CONFIG)
    if (NOT pybind11_FOUND)
        add_subdirectory(dep/pybind11)
    endif ()

    pybind11_add_module(pyaibum_core
            python/PyAibum.cpp
            )
    target_link_libraries(pyaibum_core PRIVATE aibum_core)

    install(TARGETS pyaibum_core LIBRARY DESTINATION lib)
    if (WIN32)
        # Install DLL dependencies
        install(CODE [[
                file(GET_RUNTIME_DEPENDENCIES
                    RESOLVED_DEPENDENCIES_VAR RES_DEPS
                    UNRESOLVED_DEPENDENCIES_VAR UNRES_DEPS
                    CONFLICTING_DEPENDENCIES_PREFIX CONFLICTING_DEPENDENCIES
                    EXECUTABLES $<TARGET_FILE:ScheduliteGTK>
                    DIRECTORIES ${CMAKE_MINGW_BIN}
                    PRE_EXCLUDE_REGEXES "api-ms-*" "ext-ms-*"
                    POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
                )

                message("\n\nFound dependencies:")
                foreach(DEP ${RES_DEPS})
                    message("${DEP}")
                    file(INSTALL
                        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
                        TYPE SHARED_LIBRARY
                        FOLLOW_SYMLINK_CHAIN
                        FILES "${DEP}"
                    )
                endforeach()

                message("\n\nNot found dependencies:")
                foreach(DEP ${UNRES_DEPS})
                    message("${DEP}")
                endforeach()
                ]]
                )
    endif ()
endif ()