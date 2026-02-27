cmake_minimum_required(VERSION 3.5.0)

# 注意：可执行文件名由 project() 的名字决定（见 cmake/build.cmake: add_executable(${PROJECT_NAME} ...)）。
# face_udp 工程需要生成名为 "face_udp" 的可执行文件。
project(face_udp C CXX)

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

message(STATUS "PROJECT_DIR: ${PROJECT_DIR}")

get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

include(${ROOT_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)

message(STATUS "ROOT_DIR: ${ROOT_DIR}")

include(${ROOT_DIR}/cmake/project.cmake)