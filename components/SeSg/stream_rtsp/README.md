# SeSg Stream RTSP（VPSS + 多通道 RTSP 一体化）

这个组件把以下事情**封装在一起**：

- RTSP 服务：基于 `cvi_rtsp`
- Video pipeline：复用 `components/sophgo/video`（内部会 Sys/VI/VPSS/VENC 初始化）
- 多通道：通过 VENC 回调把各通道编码数据写入对应 RTSP session
- VPSS：可选调用 `sesg_vpss_init()` / `sesg_vpss_deinit()`（默认不自动 init，避免重复初始化）
- OSD/RGN（可选）：在设备端通过 RGN 把推理结果叠加到推流画面（硬件混合，低 CPU/低内存）

对外你只需要：

- 组一个通道数组（每个通道：格式/分辨率/fps/path）
- 调用 `SesgRtspVpssStreamer::start()` 一键启动

## 头文件

```cpp
#include <sesg/stream_rtsp.h>
#include <unistd.h>
```

## CMake

在你的 solution `CMakeLists.txt` 里把组件加到 `PRIVATE_REQUIREDS`：

```cmake
component_register(
  COMPONENT_NAME your_app
  SRCS ${SOURCES}
  PRIVATE_REQUIREDS stream_rtsp
)

# 可选：启用 SG2002/CV18xx 的 RGN/OSD 叠加（需要 SDK 头文件/库）
# set(SESG_STREAM_RTSP_ENABLE_RGN ON)
```

## 使用步骤（示例：CH0/CH1/CH2 多通道 RTSP）

```cpp
// 只需要 include 这个头：
// #include <sesg/stream_rtsp.h>

sesg::stream_rtsp::SesgRtspVpssStreamer streamer;

sesg::stream_rtsp::ServerConfig server;
server.port = 8554;
// 如果不指定 ChannelConfig::path，则默认会用 ch0/ch1/ch2；
// 也可以设置 default_path_prefix，例如 "live" -> live0/live1/live2。
// server.default_path_prefix = "live";

std::vector<sesg::stream_rtsp::ChannelConfig> channels;

// CH0：H264
channels.push_back({
  .ch = VIDEO_CH0,
  .format = VIDEO_FORMAT_H264,
  .width = 1920,
  .height = 1080,
  .fps = 25,
  .path = "ch0",
  .enable_rgn_overlay = false,
});

// CH1：H264
channels.push_back({
  .ch = VIDEO_CH1,
  .format = VIDEO_FORMAT_H264,
  .width = 1280,
  .height = 720,
  .fps = 25,
  .path = "ch1",
  .enable_rgn_overlay = false,
});

// CH2：H265
channels.push_back({
  .ch = VIDEO_CH2,
  .format = VIDEO_FORMAT_H265,
  .width = 1280,
  .height = 720,
  .fps = 25,
  .path = "ch2",
  .enable_rgn_overlay = false,
});

// init_vpss=false：默认不自动 init（因为 video.startVideo() 已经会 init vpss，且 init 不是幂等的）
if (!streamer.start(server, channels, /*init_vpss=*/false)) {
  // error
}

// 访问：
// rtsp://<device-ip>:8554/ch0
// rtsp://<device-ip>:8554/ch1
// rtsp://<device-ip>:8554/ch2
```

## 最小示例（单通道，推荐最低成本跑通）

```cpp
#include <sesg/stream_rtsp.h>

int main() {
  sesg::stream_rtsp::SesgRtspVpssStreamer streamer;

  sesg::stream_rtsp::ServerConfig server;
  server.port = 8554;

  std::vector<sesg::stream_rtsp::ChannelConfig> channels;
  channels.push_back({
    .ch = VIDEO_CH2,
    .format = VIDEO_FORMAT_H264,
    .width = 1280,
    .height = 720,
    .fps = 25,
    .path = "live",
    .enable_rgn_overlay = false,
  });

  if (!streamer.start(server, channels, /*init_vpss=*/false)) {
    return 1;
  }

  // 用 VLC/ffplay 拉流：rtsp://<device-ip>:8554/live
  while (true) {
    sleep(1);
  }

  return 0;
}
```

### 退出时停止

```cpp
streamer.stop(/*deinit_vpss=*/false);
```

## FAQ / 注意事项

- **VPSS 要不要在这里 init？**
  - `app_ipcam_Vpss_Init()` 不是幂等的，重复 init 可能失败。
  - 所以 streamer 默认不自动 init；你确认系统未 init 时，再把 `start(..., init_vpss=true)` 打开。

- **这个组件会做什么初始化？**
  - `start()` 内部会调用 `initVideo()`、`setupVideo()`、`startVideo()`；这条链路会初始化 Sys/VI/VPSS/VENC。
  - 如果你的 solution 已经在别处管理了 `components/sophgo/video` 这条链路，请不要同时再用本组件。

## 设备端 OSD/RGN 叠加（SG2002/CV18xx）

SG2002（以及同系列 CV181x/CV180x）在 MPP 架构里提供 **RGN (Region)** 模块，用于 OSD、隐私遮挡等。
RGN 的混合（blending）是硬件完成的：CPU 主要负责准备位图/参数，因此适合把推理结果叠加到 RTSP 推流画面。

### 常见 RGN 类型

- `OVERLAY`：OSD 叠加（位图/透明度）
- `COVER`：纯色遮挡（Privacy Mask）
- `MOSAIC`：马赛克
- `LINE`：画线

### 核心 API（SDK）

SDK 中通常以 `CVI_RGN_` 开头：

- `CVI_RGN_Create` / `CVI_RGN_Destroy`
- `CVI_RGN_SetAttr` / `CVI_RGN_GetAttr`
- `CVI_RGN_SetBitMap`
- `CVI_RGN_AttachToChn` / `CVI_RGN_DetachFromChn`

### stream_rtsp 里的使用方式（模型无关）

1) 启动时对需要叠加的通道打开 `enable_rgn_overlay=true`。

2) 推理线程里把结果转换成 `OverlayRect`（归一化坐标：左上角 + 宽高），调用 `updateOverlayRects()`。

```cpp
sesg::stream_rtsp::ChannelConfig ch2;
ch2.ch = VIDEO_CH2;
ch2.format = VIDEO_FORMAT_H264;
ch2.width = 1280;
ch2.height = 720;
ch2.fps = 25;
ch2.path = "live";
ch2.enable_rgn_overlay = true;

// ... streamer.start/startAttached ...

// 推理输出（示意）：boxes 为归一化中心点 (cx,cy,w,h)
std::vector<sesg::stream_rtsp::OverlayRect> rects;
for (auto& b : boxes) {
  sesg::stream_rtsp::OverlayRect r;
  r.x = b.cx - b.w * 0.5f;
  r.y = b.cy - b.h * 0.5f;
  r.w = b.w;
  r.h = b.h;
  r.argb = 0xFFFF0000u; // red
  r.thickness = 2;
  rects.push_back(r);
}
streamer.updateOverlayRects(VIDEO_CH2, rects);
```

### 构建开关

- 需要 SDK 头文件 `cvi_region.h` 或 `cvi_rgn.h`。
- CMake 打开：`SESG_STREAM_RTSP_ENABLE_RGN=ON`。
- 如果缺少 RGN 头文件：会安全退化为 no-op（不影响 RTSP 推流）。

