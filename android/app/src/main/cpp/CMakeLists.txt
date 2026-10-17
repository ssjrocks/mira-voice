cmake_minimum_required(VERSION 3.22.1)
project(miravoice C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# workspace root: android/app/src/main/cpp -> ../../../../..
get_filename_component(ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../.. ABSOLUTE)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
    set(GGML_OPENMP ON CACHE BOOL "" FORCE)
else()
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(${ROOT}/upstream/llama.cpp build-llama)

# speech recognition (whisper.cpp) shares llama.cpp's ggml
set(WHISPER_DIR ${ROOT}/upstream/whisper.cpp)
add_library(miravoice SHARED mira_jni.cpp ${ROOT}/native/mira_engine.cpp ${ROOT}/native/mira_asr.cpp
        ${WHISPER_DIR}/src/whisper.cpp ${WHISPER_DIR}/src/aneforge/whisper-aneforge.cpp)
target_include_directories(miravoice PRIVATE ${ROOT}/native ${WHISPER_DIR}/include ${WHISPER_DIR}/src)
target_compile_definitions(miravoice PRIVATE WHISPER_VERSION="1.9.5")
target_compile_options(miravoice PRIVATE -O3)
# fast-math only for the engine's sampler/head; whisper.cpp relies on infinities when masking tokens
set_source_files_properties(${ROOT}/native/mira_engine.cpp PROPERTIES COMPILE_OPTIONS "-ffast-math")
target_link_libraries(miravoice llama ggml android log)
