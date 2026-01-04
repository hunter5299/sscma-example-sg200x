# cls_test 原始测试工具（保留 yolov8_postprocess）
file(GLOB cls_test_srcs ${CMAKE_CURRENT_LIST_DIR}/main.cpp ${CMAKE_CURRENT_LIST_DIR}/yolov8_postprocess.cpp)

set(incs ${CMAKE_CURRENT_LIST_DIR})

component_register(
    COMPONENT_NAME main
    SRCS ${cls_test_srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS sscma-micro
)
