cmake_minimum_required(VERSION 3.5.0)

# 注意：可执行文件名由 project() 的名字决定（见 cmake/build.cmake: add_executable(${PROJECT_NAME} ...)）。
# face_rtsp 工程需要生成名为 "face_rtsp" 的可执行文件。
project(face_rtsp C CXX)

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

message(STATUS "PROJECT_DIR: ${PROJECT_DIR}")

get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

include(${ROOT_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)

message(STATUS "ROOT_DIR: ${ROOT_DIR}")

# Enable SG2002 RGN/OSD overlay support in stream_rtsp
set(SESG_STREAM_RTSP_ENABLE_RGN ON CACHE BOOL "Enable RGN overlay" FORCE)

include(${ROOT_DIR}/cmake/project.cmake)