# Desktop build of the MiraTTS engine + test CLI (the Android app builds the same sources via its own CMake).
cmake_minimum_required(VERSION 3.22)
project(mira_native CXX C)
set(CMAKE_CXX_STANDARD 17)

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../upstream/llama.cpp)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(NOT DEFINED CACHE{BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
endif()
add_subdirectory(${LLAMA_DIR} llama.cpp)

add_library(mira_engine STATIC mira_engine.cpp)
target_link_libraries(mira_engine PUBLIC llama)
target_include_directories(mira_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(mira_cli mira_cli.cpp)
target_link_libraries(mira_cli PRIVATE mira_engine)
if(NOT MSVC)
  target_compile_options(mira_engine PRIVATE -O3 -ffast-math)
endif()

# Speech recognition (whisper.cpp) built against llama.cpp's ggml, so the app ships one copy of ggml.
set(WHISPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../upstream/whisper.cpp)
add_library(mira_asr STATIC mira_asr.cpp ${WHISPER_DIR}/src/whisper.cpp ${WHISPER_DIR}/src/aneforge/whisper-aneforge.cpp)
target_include_directories(mira_asr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${WHISPER_DIR}/include PRIVATE ${WHISPER_DIR}/src)
target_compile_definitions(mira_asr PRIVATE WHISPER_VERSION="1.9.5" WHISPER_SHARED_EXPORT= )
target_link_libraries(mira_asr PUBLIC ggml)

add_executable(mira_asr_cli mira_asr_cli.cpp)
target_link_libraries(mira_asr_cli PRIVATE mira_asr)
