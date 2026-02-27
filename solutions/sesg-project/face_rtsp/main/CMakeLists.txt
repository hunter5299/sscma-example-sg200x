set(srcs
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detector_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/age_gender_race_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/emotion_runner.cpp
)

set(incs ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/node
)

component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS sscma-micro stream_rtsp sophgo
)