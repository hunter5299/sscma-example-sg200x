# SeSg Stream UDP（VPSS + JPEG + UDP 一体化）

这个组件把以下事情**封装在一起**：

- UDP 发送：复用 `components/SeSg/udp_service` 的协议与 `SesgUDPSender`
- JPEG 拉流线程：复用 `solutions/udp_face` 的实践，避免主循环因 `retrieveFrame(JPEG)` 被阻塞
- VPSS：可选调用 `sesg_vpss_init()` / `sesg_vpss_deinit()`（默认不自动 init，避免重复初始化）

对外你只需要：

- 给相机配置一个额外通道（例如 CH2/CH3）输出 JPEG
- 在主循环推理后把检测结果交给 `SesgJpegUdpStreamer::sendLatest()`

## 头文件

```cpp
#include <sesg/stream_udp.h>
```

## CMake

在你的 solution `CMakeLists.txt` 里把组件加到 `PRIVATE_REQUIREDS`：

```cmake
component_register(
  COMPONENT_NAME your_app
  SRCS ${SOURCES}
  PRIVATE_REQUIREDS stream_udp
)
```

## 使用步骤（推荐：CH0 推理，CH2/CH3 UDP）

### 1) 配置相机通道（在 startStream 前）

```cpp
// CH0：推理通道，通常是 RGB888
// ...（按你现有逻辑）

// CH2：JPEG 通道，用于 UDP 推送
sesg::stream_udp::JpegChannelConfig jpeg_cfg;
jpeg_cfg.ch = 2;          // 你希望给 UDP 用的通道（CH2/CH3 均可）
jpeg_cfg.width = 320;
jpeg_cfg.height = 240;
jpeg_cfg.fps = 10;

sesg::stream_udp::SesgJpegUdpStreamer::configureJpegChannel(*camera, jpeg_cfg);

camera->startStream(Camera::StreamMode::kRefreshOnReturn);
```

建议 JPEG 分辨率尽量让单帧接近/小于 60KB，以降低 UDP 分包丢包导致的花屏/解码失败概率。

### 2) 启动 streamer（内部启动 UDP sender + JPEG 拉流线程）

```cpp
sesg::stream_udp::SesgJpegUdpStreamer streamer("192.168.1.100", 3456);

// init_vpss=false：默认不自动 init（因为 VPSS init 不是幂等的）
// 如果你的系统没有其它地方初始化 VPSS，可以把 init_vpss 设为 true。
if (!streamer.start(camera_shared, jpeg_cfg, /*init_vpss=*/false)) {
    // error
}
```

### 3) 主循环中发送（与推理结果同步）

```cpp
struct MyDet {
    float x, y, w, h, score;
    int32_t cls;
};

std::vector<MyDet> dets = ...;  // 你的推理结果

constexpr uint32_t UDP_MAGIC = 0x55445031; // 示例：'UDP1'，你也可以沿用项目里已有 magic
streamer.sendLatest(UDP_MAGIC, dets);

// 可选：streamer.getSendFPS();
```

### 4) 退出时停止

```cpp
streamer.stop(/*deinit_vpss=*/false);
```

## FAQ / 注意事项

- **为什么用 JPEG 通道发 UDP？**
  - `udp_service` 的 `sendRgb888Physical` 语义是“物理地址 + mmap 拷贝”，而很多 solution 为了 CPU 访问会用虚拟地址缓存（如 memcpy 到 CPU buffer），两者容易混淆。JPEG 路径更清晰：从 VENC 直接拿到压缩数据后发送。

- **VPSS 要不要在这里 init？**
  - `app_ipcam_Vpss_Init()` 不是幂等的，重复 init 可能失败。
  - 所以 streamer 默认不自动 init；你确认系统未 init 时，再把 `start(..., init_vpss=true)` 打开。
