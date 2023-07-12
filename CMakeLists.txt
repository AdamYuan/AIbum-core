cmake_minimum_required(VERSION 3.10)
project(aibum-core)

set(CMAKE_CXX_STANDARD 20)

option(AIBUM_WASM_ONLY "build WASM only" ON)
option(AIBUM_WASM_SIMD "build WASM with SIMD enabled" ON)
option(AIBUM_WASM_THREADS "build WASM with Threads enabled" ON)

option(AIBUM_TEST "build tests" ON)

add_library(aibum_core STATIC
		src/SCRFD.cpp
		src/FaceNet.cpp
		src/ImageNet.cpp
		src/StyleTransfer.cpp
		)
target_include_directories(aibum_core PUBLIC include PRIVATE models/include)

set(NCNN_BUILD_BENCHMARK OFF)
set(NCNN_BUILD_EXAMPLES OFF)
set(NCNN_BUILD_TOOLS OFF)
set(NCNN_INSTALL_SDK OFF)
set(NCNN_SIMPLEOCV OFF)
set(NCNN_INT8 OFF)
if (AIBUM_WASM_ONLY)
	set(NCNN_DISABLE_RTTI OFF)
	set(NCNN_DISABLE_EXCEPTION ON)
	set(NCNN_STDIO OFF)
	set(NCNN_STRING OFF)
	set(NCNN_RUNTIME_CPU OFF)
	set(NCNN_AVX2 OFF)
	set(NCNN_AVX OFF)

	if (AIBUM_WASM_SIMD)
		set(NCNN_SSE2 ON)
	else ()
		set(NCNN_SSE2 OFF)
	endif ()

	if (AIBUM_WASM_THREADS)
		set(NCNN_THREADS ON)
		set(NCNN_OPENMP ON)
		set(NCNN_SIMPLEOMP ON)
	else ()
		set(NCNN_THREADS OFF)
		set(NCNN_OPENMP OFF)
		set(NCNN_SIMPLEOMP OFF)
	endif ()
endif ()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(dep/ncnn)

target_link_libraries(aibum_core PUBLIC ncnn)

if (NOT AIBUM_WASM_ONLY)
	if (AIBUM_TEST)
		find_package(OpenCV REQUIRED)

		add_executable(aibum_class_test test/class.cpp)
		target_link_libraries(aibum_class_test PRIVATE aibum_core ${OpenCV_LIBS})

		add_executable(aibum_face_test test/face.cpp)
		target_link_libraries(aibum_face_test PRIVATE aibum_core ${OpenCV_LIBS})

		add_executable(aibum_face_cmp_test test/face_cmp.cpp)
		target_link_libraries(aibum_face_cmp_test PRIVATE aibum_core ${OpenCV_LIBS})

		add_executable(aibum_style_test test/style.cpp)
		target_link_libraries(aibum_style_test PRIVATE aibum_core ${OpenCV_LIBS})
	endif ()
else ()
	target_compile_definitions(aibum_core PRIVATE -DAIBUM_CORE_WASM)
	target_compile_options(aibum_core PRIVATE
			-fno-exceptions
			-flto
			)

	add_executable(aibum_core_wasm wasm/aibum_core_wasm.cpp)
	target_include_directories(aibum_core_wasm PRIVATE wasm/dep/stb)
	target_link_libraries(aibum_core_wasm PRIVATE aibum_core)
	target_compile_options(aibum_core_wasm PRIVATE
			-fno-exceptions
			-flto
			)
	target_link_options(aibum_core_wasm PRIVATE
			--bind
			-sMODULARIZE=1
			-sEXPORT_ES6=1
			-sDISABLE_EXCEPTION_CATCHING=1
			-sEXPORT_NAME=loadAIbumCore
			-sALLOW_MEMORY_GROWTH
			-flto
			-sTOTAL_MEMORY=512MB
			# -sEXIT_RUNTIME=1
			-sASYNCIFY
			# -sASYNCIFY_ONLY=["WASMImageNet::getTags\(WASMImage const&, int\) const","WASMFaceNet::getFaces\(WASMImage const&\) const"]
			)

	if (AIBUM_WASM_SIMD)
		target_compile_options(aibum_core PRIVATE
				-msse2
				-msse
				-msimd128
				)
		target_compile_options(aibum_core_wasm PRIVATE
				-msse2
				-msse
				-msimd128
				)
		target_compile_definitions(aibum_core_wasm PRIVATE -DAIBUM_WASM_SIMD)
	endif ()

	if (AIBUM_WASM_THREADS)
		target_link_options(aibum_core_wasm PRIVATE
				-sENVIRONMENT=web,worker
				-sUSE_PTHREADS=1
				-sPTHREAD_POOL_SIZE=15
				)
		target_compile_definitions(aibum_core_wasm PRIVATE -DAIBUM_WASM_THREADS)
	else ()
		target_link_options(aibum_core_wasm PRIVATE
				-sENVIRONMENT=web
				)
	endif ()

	install(FILES
			"$<TARGET_FILE_DIR:aibum_core_wasm>/aibum_core_wasm.js"
			"$<TARGET_FILE_DIR:aibum_core_wasm>/aibum_core_wasm.wasm"
			DESTINATION .)
	if (AIBUM_WASM_THREADS)
		install(FILES
				"$<TARGET_FILE_DIR:aibum_core_wasm>/aibum_core_wasm.worker.js"
				DESTINATION .)
	endif ()
endif ()
