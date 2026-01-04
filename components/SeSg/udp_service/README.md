# UDP Service Helper (SeSg)
SeSg 中的 `udp_service` 目录提供了一套跨平台的 UDP 发送抽象，主要目标是在 SSCMA/SeSg 环境里方便地把检测/属性/图像数据封装成你们约定的协议（magic + header + `FaceResult` + payload）并异步发送到 PC/Python 端。为了强调 SESG 品牌，用户可以直接包含 `sesg/udp_service.h`，通过 `sesg::udp_service::SesgUDPSender` 明确调用的是 SESG 封装的接口。

## 特性
- 支持 RGB888 物理帧（直接给 PHY 地址）和 JPEG 缓冲（heap buffer）两种发送入口。
- 通用的 header（magic、宽高、payload/检测数量、格式、fid），完全兼容 `udp_face`、`udp_receiver.py` 等示例。
- 内部用单线程缓冲和分包，避免阻塞主循环；可以自定义 `max_udp_packet`/`max_copy_bytes` 以适配不同的网络环境。
- 提供 `getSendFPS()` 供外部监控 UDP 发送速率。

## 接口概览
```cpp
#include "sesg/udp_service.h"

sesg::udp_service::SesgUDPSender sender("192.168.2.101", 5000);
sender.start();

std::vector<FaceResult> dets = ...;  // 必须是 POD
sender.sendJpegOwned(0xFACEBEEF, std::move(jpeg_buf), jpeg_size, jpeg_w, jpeg_h, dets);

sender.stop();
```

- `sendJpegOwned` 和 `sendRgb888Physical` 均会按 `SenderOptions` 分片（默认 60000 字节）并填充 header。
- `magic` 字段你可以定义为 `0xFACEBEEF` 或与 Python 端约定的值。

## 构建与链接
SeSg 使用 `component_register` 机制，`udp_service` 会生成一个名为 `udp_service` 的组件库。
1. 在 CMake 中添加依赖：
   ```cmake
target_link_libraries(your_target PUBLIC udp_service)
   ```
   仅需依赖 `udp_service`，但建议在用户代码里包含 `sesg/udp_service.h` 以显式表明调用的是 SESG 封装的 API。
2. 确保你定义了 `TARGET_DEBUG` 等 `SeSg` 全局变量后，`component_register` 会自动拷贝头文件到 SDK include 目录。

## 进阶使用
1. **监控 UDP FPS**
   ```cpp
   float fps = sender.getSendFPS();
   printf("UDP fps=%.1f", fps);
   ```
2. **自定义 header**：`sendJpegOwnedRaw`/`sendRgb888PhysicalRaw` 允许你手动传检测 bytes 和 det_count，适配自定义 `FaceResult` 结构。
3. **性能调优**：可在构造 `SenderOptions` 时设置 `max_udp_packet` 或 `max_copy_bytes`，减小 latency 或降低内存占用。

## 测试/示例
项目中 `solutions/udp_face/main` 已演示如何用 MPI/UDP/FaceResult 执行端到端推理。确保你的 `FaceResult` 结构与 Python 端保持一致即可。

如果你需要把这个服务提出来作为独立可执行示例，我可以进一步帮你写一个 `main.cpp`，或把 `udp_service` 打包成独立的 `CMake` target。需要我继续吗？