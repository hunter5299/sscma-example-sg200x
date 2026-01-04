file(GLOB_RECURSE SESG_STREAM_UDP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

component_register(
    COMPONENT_NAME stream_udp
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    SRCS ${SESG_STREAM_UDP_SOURCES}
    PRIVATE_REQUIREDS udp_service sscma-micro
)

# Optional: link SeSg VPSS wrapper if you want streamer.start(init_vpss=true)
option(SESG_STREAM_UDP_ENABLE_VPSS "Enable sesg_vpss_init/deinit inside stream_udp" OFF)
if(SESG_STREAM_UDP_ENABLE_VPSS)
    target_compile_definitions(stream_udp PRIVATE SESG_STREAM_UDP_ENABLE_VPSS=1)
    target_link_libraries(stream_udp PRIVATE sesg_vpss)
endif()
