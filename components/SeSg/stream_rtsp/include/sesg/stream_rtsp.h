#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <video.h>

// VPSS 封装可选：默认不启用，避免把 sophgo/VPSS 链路强制拉进所有使用者。
// 如需启用，请在构建时打开 CMake 选项：SESG_STREAM_RTSP_ENABLE_VPSS=ON
#if defined(SESG_STREAM_RTSP_ENABLE_VPSS)
#include <sesg/vpss.h>
#endif

namespace sesg::stream_rtsp {

struct ChannelConfig {
    video_ch_index_t ch = VIDEO_CH2;
    video_format_t format = VIDEO_FORMAT_H264;
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint8_t fps = 25;

    // 可选：设备端 RGN/OSD 叠加（硬件混合，不走软件重编码）。
    // - 需要 SDK 提供 CVI_RGN_* 相关头文件/库。
    // - 不启用时不影响纯推流。
    bool enable_rgn_overlay = false;

    // 对应 RTSP path：rtsp://<ip>:<port>/<path>
    // 为空时默认使用："ch0/ch1/ch2"。
    std::string path;
};

struct ServerConfig {
    int port = 8554;

    // 当 ChannelConfig::path 为空时生效。
    // 例如 prefix="live" 时，将生成 live0/live1/live2（按通道号）。
    std::string default_path_prefix;
};

// ========== Overlay / OSD（可选）==========
// 设计目标：让不同模型（YOLO/pose/seg/...）只需要把结果转换成“绘制原语”，
// streamer 负责在设备端通过 RGN/OSD 叠加到视频通道上。

struct OverlayRect {
    // 归一化坐标（左上角 + 宽高），范围建议 [0,1]。
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;

    // ARGB8888（A 在高 8 bit）。默认：不透明红色。
    uint32_t argb = 0xFFFF0000u;

    // 线宽（像素）。
    uint16_t thickness = 2;
};

// RTSP + VPSS 多通道一体化封装（基于 sophgo/video + cvi_rtsp）。
// - 内部调用 initVideo/setupVideo/registerVideoFrameHandler/startVideo
// - 内部创建 RTSP server，并把对应 VENC 输出写入各 session
class SesgRtspVpssStreamer {
public:
    SesgRtspVpssStreamer();
    ~SesgRtspVpssStreamer();

    SesgRtspVpssStreamer(const SesgRtspVpssStreamer&) = delete;
    SesgRtspVpssStreamer& operator=(const SesgRtspVpssStreamer&) = delete;

    // 一键启动：配置多通道 + 启动 RTSP server + startVideo。
    // 注意：sophgo/video 的 init/start/deinit 不是幂等的；如果系统里已有其它模块在管理 video 链路，请不要同时使用。
    bool start(const ServerConfig& server_cfg, const std::vector<ChannelConfig>& channels,
               bool init_vpss = false);

    // 附着模式：复用外部已管理的 VI/VPSS/VENC 管线（例如 ma::CameraSG200X）。
    // - 内部仍会调用 setupVideo() 以确保对应 VENC 通道启用/尺寸正确。
    // - 不会调用 initVideo()/startVideo()/deinitVideo()，避免重复初始化。
    // - 默认使用 consumer_index=1 注册 VENC 数据回调，避免覆盖外部模块通常使用的 index=0。
    bool startAttached(const ServerConfig& server_cfg, const std::vector<ChannelConfig>& channels,
                       int consumer_index = 1);

    // 停止 RTSP + deinitVideo。
    void stop(bool deinit_vpss = false);

    // 停止 RTSP（不 deinitVideo）。
    // 会尝试把已注册的 consumer_index 对应 handler 清空。
    void stopAttached();

    bool isRunning() const;

    // 更新指定通道的 overlay（如果该通道启用了 enable_rgn_overlay）。
    // - rect 坐标为归一化坐标（相对该通道输出分辨率）。
    // - 实际叠加方式取决于平台是否提供 RGN/OSD（无则安全退化为 no-op 并返回 false）。
    bool updateOverlayRects(video_ch_index_t ch, const std::vector<OverlayRect>& rects);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sesg::stream_rtsp
