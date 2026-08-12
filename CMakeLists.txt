cmake_minimum_required(VERSION 3.10)
project(SurveilanceCam CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

# nlohmann/json dipakai oleh APIController untuk serialize/parse pesan
# WebSocket (deteksi AI, command config, dsb).
#   sudo apt install nlohmann-json3-dev
find_package(nlohmann_json REQUIRED)

# Modul GStreamer yang dibutuhkan:
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-rtsp-server-1.0
    gstreamer-pbutils-1.0
)

# ---- Lokasi TensorRT ----
# Kalau TensorRT terinstall via .deb NVIDIA (biasanya di Jetson / JetPack),
# header & lib sudah ada di /usr/include & /usr/lib/<arch>-linux-gnu, tidak perlu diubah.
# Kalau install manual (tar.gz dari NVIDIA Developer), set path lewat:
#   cmake -DTENSORRT_DIR=/path/ke/TensorRT-x.y.z ..
set(TENSORRT_DIR "" CACHE PATH "Path ke instalasi TensorRT manual (kosongkan kalau pakai JetPack/apt)")

# ---- Lokasi websocketpp ----
# Header-only, dipakai oleh APIController untuk WebSocket server.
# Default: deps/websocketapp (relatif ke root project).
# Kalau lokasinya beda, override lewat:
#   cmake -DWEBSOCKETPP_INCLUDE_DIR=/path/lain ..
set(WEBSOCKETPP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/deps/websocketapp" CACHE PATH "Path to websocketpp headers")

if(TENSORRT_DIR)
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_DIR}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_DIR}/lib)
else()
    set(TENSORRT_INCLUDE_DIR /usr/include)
    set(TENSORRT_LIB_DIR /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE})
endif()

find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_LIB_DIR})
find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TENSORRT_LIB_DIR})

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "Tidak menemukan libnvinfer. Set -DTENSORRT_DIR=... kalau TensorRT terinstall manual, "
                         "atau pastikan sudah install TensorRT (misal via apt/JetPack).")
endif()

include_directories(
    ${GST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${WEBSOCKETPP_INCLUDE_DIR}
)

link_directories(${GST_LIBRARY_DIRS} ${TENSORRT_LIB_DIR})

add_executable(SurveilanceCam
    src/main.cpp
    src/YoloDetector.cpp
    src/rtspServer.cpp
    src/APIController.cpp
    src/ConfigManager.cpp
    src/SourceHandler.cpp
    src/CrowdCounting.cpp
)

target_compile_options(SurveilanceCam PUBLIC ${GST_CFLAGS_OTHER})

target_link_libraries(SurveilanceCam
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    ${NVINFER_LIB}
    CUDA::cudart
    Boost::system
    Threads::Threads
    nlohmann_json::nlohmann_json
    pthread
)