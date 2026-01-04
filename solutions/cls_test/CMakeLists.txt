cmake_minimum_required(VERSION 3.5.0)

project(cls_test C CXX)

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

message(STATUS "PROJECT_DIR: ${PROJECT_DIR}")

get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

include(${ROOT_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)

message(STATUS "ROOT_DIR: ${ROOT_DIR}")

include(${ROOT_DIR}/cmake/project.cmake)

# 额外生成 camera_test 可执行文件
add_executable(camera_test ${CMAKE_CURRENT_LIST_DIR}/main/camera_test.cpp)
target_include_directories(camera_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main)
target_link_libraries(camera_test PRIVATE sscma-micro sophgo main)
