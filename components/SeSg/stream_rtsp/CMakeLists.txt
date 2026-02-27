file(GLOB_RECURSE SESG_STREAM_RTSP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

component_register(
    COMPONENT_NAME stream_rtsp
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    SRCS ${SESG_STREAM_RTSP_SOURCES}
    PRIVATE_REQUIREDS sophgo cvi_rtsp
)

# Optional: enable SG2002/CV18xx RGN/OSD overlay support (requires SDK headers: cvi_region.h/cvi_rgn.h)
option(SESG_STREAM_RTSP_ENABLE_RGN "Enable SG2002 RGN/OSD overlay support inside stream_rtsp" OFF)
if(SESG_STREAM_RTSP_ENABLE_RGN)
    target_compile_definitions(stream_rtsp PRIVATE SESG_STREAM_RTSP_ENABLE_RGN=1)
endif()

# Optional: link SeSg VPSS wrapper if you want streamer.start(init_vpss=true)
option(SESG_STREAM_RTSP_ENABLE_VPSS "Enable sesg_vpss_init/deinit inside stream_rtsp" OFF)
if(SESG_STREAM_RTSP_ENABLE_VPSS)
    target_compile_definitions(stream_rtsp PRIVATE SESG_STREAM_RTSP_ENABLE_VPSS=1)
    target_link_libraries(stream_rtsp PRIVATE sesg_vpss)
endif()
