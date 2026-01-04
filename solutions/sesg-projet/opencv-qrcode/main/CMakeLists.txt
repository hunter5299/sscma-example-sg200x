file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

set(incs ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS sscma-micro stream_udp
    REQUIREDS
        # OpenCV imgcodecs 仍沿用 SDK 自带（用于 imread）
        opencv_imgcodecs
        # 触发引入 components/SeSg/opencv_objdetect 的头文件路径
        opencv_objdetect
        # 强制使用 SeSg/opencv_objdetect/lib 下自编的 OpenCV 子库，避免与 SDK OpenCV core 混用
        ${ROOT_DIR}/components/SeSg/opencv_objdetect/lib/libopencv_objdetect.so.4.5.0
        ${ROOT_DIR}/components/SeSg/opencv_objdetect/lib/libopencv_calib3d.so.4.5.0
        ${ROOT_DIR}/components/SeSg/opencv_objdetect/lib/libopencv_features2d.so.4.5.0
        ${ROOT_DIR}/components/SeSg/opencv_objdetect/lib/libopencv_flann.so.4.5.0
        ${ROOT_DIR}/components/SeSg/opencv_objdetect/lib/libopencv_imgproc.so.4.5.0
        ${ROOT_DIR}/components/SeSg/opencv_objdetect/lib/libopencv_core.so.4.5.0
        z
)
