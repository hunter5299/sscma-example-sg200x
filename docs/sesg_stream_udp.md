# SeSg 一体化：CH2/CH3 JPEG + UDP（stream_udp）开发者文档

本文面向在 SG200X / ReCamera 平台上开发应用的同学，目标是把“额外通道（CH2/CH3）的视频帧”通过 UDP 推送到上位机，同时不影响主循环推理性能。

## 你将得到什么

- 一套“SeSg 特色”的对外 API：`#include <sesg/stream_udp.h>`
- 一个更稳的推荐链路：**CH0 推理（RGB888） + CH2/CH3 JPEG（VENC） + UDP 推送**
- 内置 JPEG 拉流线程：避免 `retrieveFrame(JPEG)` 阻塞主循环

## 组件位置

- 组件代码：`components/SeSg/stream_udp`
- 对外头文件：`components/SeSg/stream_udp/include/sesg/stream_udp.h`
- 依赖组件：
  - `components/SeSg/udp_service`（UDP 协议与 sender）
  - `components/SeSg/vpss`（可选 vpss init/deinit）

## 设计要点（为什么这样封装）

1) **通道职责明确**
- CH0：推理/算法（RGB888）
- CH2/CH3：仅用于网络推送（JPEG）

2) **避免“物理/虚拟地址”语义冲突**
- `udp_service::sendRgb888Physical` 需要物理地址，并在内部 `CVI_SYS_Mmap` 后拷贝。
- 很多业务为了 CPU 访问会把帧拷贝到 CPU buffer（虚拟地址），如果继续走 physical 发送，容易踩坑。
- 因此推荐走 **JPEG 通道**：数据天然是 CPU buffer、大小更小、网络更友好。

3) **避免主循环阻塞**
- `retrieveFrame(JPEG)` 常见实现会“按 fps 等待”，在主循环里每帧拉 JPEG 会把整体 FPS 拉低。
- streamer 内置 JPEG 拉流线程，主循环只取最新的一帧发送（丢旧帧换流畅）。

## 快速开始

### 1) 在你的 solution 里链接组件

在 solution 的 `CMakeLists.txt` 里，把 `sesg_stream_udp` 加到 `PRIVATE_REQUIREDS`：

```cmake
component_register(
  COMPONENT_NAME your_app
  SRCS ${SOURCES}
  PRIVATE_REQUIREDS stream_udp
)
```

### 2) 配置相机通道（CH2 或 CH3 输出 JPEG）

关键点：**务必在 `camera->startStream()` 之前配置**。

```cpp
#include <sesg/stream_udp.h>

// ... 你创建 camera 并 camera->init(0) 之后

// 例如：CH0 配置为推理 RGB888（按你现有逻辑）

// 再配置 CH2 为 JPEG，用于 UDP
sesg::stream_udp::JpegChannelConfig jpeg_cfg;
jpeg_cfg.ch = 2;
jpeg_cfg.width = 320;
jpeg_cfg.height = 240;
jpeg_cfg.fps = 10;

sesg::stream_udp::SesgJpegUdpStreamer::configureJpegChannel(*camera, jpeg_cfg);

camera->startStream(Camera::StreamMode::kRefreshOnReturn);
```

建议：
- JPEG 分辨率优先从 `320x240` / `416x416` 这类开始试
- fps 不要设置得高于主循环的实际处理能力，否则 VENC 内部可能出现缓存积压并丢帧

### 3) 启动 streamer

```cpp
sesg::stream_udp::SesgJpegUdpStreamer streamer("192.168.1.100", 3456);

// init_vpss 默认 false（VPSS init 不是幂等的，重复 init 可能失败）
// 如果你确认系统没有其它地方 init VPSS，可以设为 true。
if (!streamer.start(camera_shared, jpeg_cfg, /*init_vpss=*/false)) {
    // handle error
}
```

### 4) 主循环中发送（与推理结果同步）

你可以定义一个“可 trivially copyable 的结构体”，直接把 bbox/score/class 等塞进去：

```cpp
struct MyDet {
  float x, y, w, h;
  float score;
  int32_t cls;
};

std::vector<MyDet> dets = ...; // 推理结果

constexpr uint32_t UDP_MAGIC = 0x55445031; // 'UDP1' 仅示例
streamer.sendLatest(UDP_MAGIC, dets);
```

或者你已经有 raw buffer（自定义协议/结构体数组），也可以用 raw 发送：

```cpp
streamer.sendLatestRaw(UDP_MAGIC, det_ptr, det_bytes, det_count);
```

### 5) 停止

```cpp
streamer.stop(/*deinit_vpss=*/false);
```

## 常见问题

### Q1：为什么我 sendLatest 经常返回 -1？
说明当前还没有拉到 JPEG 帧（或刚启动、网络线程还没跑起来）。
- 确认 CH2/CH3 已配置为 JPEG
- 确认 `camera->startStream()` 已调用
- 确认 `streamer.start()` 在 startStream 之后调用

### Q2：VPSS 初始化到底应该放哪里？
- `app_ipcam_Vpss_Init()` 不是幂等的，重复 init 可能失败。
- 如果你使用的上层框架已经启动了视频链路（Sys/VI/VPSS/VENC），不要在 streamer 里再 init。
- 如果你是“裸用 camera”且发现 JPEG/VENC 不工作，再尝试 `start(..., init_vpss=true)`。

### Q3：CH2/CH3 和 CH1 有什么区别？
本质是“你愿意把哪个通道留给谁”。
- 如果 CH1 已被别的业务占用（比如 RTSP/显示），那就用 CH2/CH3。
- streamer 的 `JpegChannelConfig::ch` 允许你自由指定。
