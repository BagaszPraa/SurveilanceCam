cmake_minimum_required(VERSION 3.14)
project(rtsp_server VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Cari paket GStreamer via pkg-config ──────────────────────────────────────
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST          REQUIRED gstreamer-1.0)
pkg_check_modules(GST_RTSP     REQUIRED gstreamer-rtsp-server-1.0)
pkg_check_modules(GST_RTSP_LIB REQUIRED gstreamer-rtsp-1.0)
pkg_check_modules(GLIB         REQUIRED glib-2.0)

# ── Executable ──────────────────────────────────────────────────────────────
add_executable(rtsp_server main.cpp)

# ── Include directories ──────────────────────────────────────────────────────
target_include_directories(rtsp_server PRIVATE
    ${GST_INCLUDE_DIRS}
    ${GST_RTSP_INCLUDE_DIRS}
    ${GST_RTSP_LIB_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
)

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(rtsp_server PRIVATE
    ${GST_LIBRARIES}
    ${GST_RTSP_LIBRARIES}
    ${GST_RTSP_LIB_LIBRARIES}
    ${GLIB_LIBRARIES}
)

# ── Compile flags ─────────────────────────────────────────────────────────────
target_compile_options(rtsp_server PRIVATE
    ${GST_CFLAGS_OTHER}
    ${GST_RTSP_CFLAGS_OTHER}
    ${GST_RTSP_LIB_CFLAGS_OTHER}
    -Wall
    -Wextra
    -O2
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS rtsp_server DESTINATION bin)

message(STATUS "GStreamer version      : ${GST_VERSION}")
message(STATUS "RTSP Server version   : ${GST_RTSP_VERSION}")
message(STATUS "RTSP Library version  : ${GST_RTSP_LIB_VERSION}")
message(STATUS "Build type            : ${CMAKE_BUILD_TYPE}")