set(srcs
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/yolov8_postprocess.cpp
)

set(incs ${CMAKE_CURRENT_LIST_DIR})

component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS sscma-micro cvi_rtsp
)
