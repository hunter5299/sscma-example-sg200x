set(SOPHGO_PLATFORM ON)

include(${ROOT_DIR}/cmake/macro.cmake)

set(CMAKE_CXX_STANDARD 17)

if(NOT "${SG200X_SDK_PATH}" STREQUAL "")
    message(STATUS "SG200X_SDK_PATH: ${SG200X_SDK_PATH}")

    if("${SYSROOT}" STREQUAL "")
        set(SYSROOT ${SG200X_SDK_PATH}/buildroot-2021.05/output/cvitek_CV181X_musl_riscv64/host/riscv64-buildroot-linux-musl/sysroot)
    endif()

    message(STATUS "SYSROOT: ${SYSROOT}")

    include_directories("${SYSROOT}/usr/include")
    link_directories("${SYSROOT}/usr/lib")

    include_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include")
    link_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/lib")
    link_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/rootfs/mnt/system/lib")

    # rtsp
    include_directories("${SG200X_SDK_PATH}/cvi_rtsp/install/include/cvi_rtsp")
    link_directories("${SG200X_SDK_PATH}/cvi_rtsp/install/lib")
    

    include_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/liveMedia")
    include_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/groupsock")
    include_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/UsageEnvironment")
    include_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/BasicUsageEnvironment")

else()
    message(WARNING "SG200X_SDK_PATH is not set")
endif()

file(GLOB _COMPONENTS_L1 LIST_DIRECTORIES true ${ROOT_DIR}/components/*)
# Allow grouping sub-components under components/SeSg/*
file(GLOB _COMPONENTS_SESG LIST_DIRECTORIES true ${ROOT_DIR}/components/SeSg/*)
set(COMPONENTS ${_COMPONENTS_L1} ${_COMPONENTS_SESG})
include(${PROJECT_DIR}/main/CMakeLists.txt)

set(INCLUDED_COMPONENT_NAMES "")

# Components can add new transitive dependencies via component_register(), so we
# iteratively include required components until the dependency set converges.
set(_changed TRUE)
while(_changed)
    set(_changed FALSE)
    foreach(component IN LISTS COMPONENTS)
        get_filename_component(component_name ${component} NAME)
        message(STATUS "component: ${component_name}")

        if(EXISTS "${component}/CMakeLists.txt" AND component_name IN_LIST REQUIREDS)
            if(NOT component_name IN_LIST INCLUDED_COMPONENT_NAMES)
                include("${component}/CMakeLists.txt")
                list(APPEND INCLUDED_COMPONENT_NAMES ${component_name})
                set(_changed TRUE)
            endif()
        endif()
    endforeach()
endwhile()

include(${ROOT_DIR}/cmake/build.cmake)

include(${ROOT_DIR}/cmake/package.cmake)
