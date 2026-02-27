# UDP 人脸分析（face_udp）

## 概述

**face_udp** 是一个示例应用程序，演示如何结合实时摄像头捕获和 AI 模型推理，使用 **SSCMA-Micro** 框架。该应用从摄像头捕获视频帧，执行人脸检测 + 属性/表情推理，并通过 UDP 推送 JPEG + 检测元数据。

仅支持 **正方形输入** 的 SSCMA 原生 Detector（BBox 输出），并提供详细性能统计。

当前数据流：
- CH0：RGB888（物理地址）用于推理
- CH1：JPEG（VENC）用于 UDP 推送（不使用 RTSP）

## 文件结构

- `main.cpp` - 主程序入口，简洁清晰
- `detector_utils.h/cpp` - 检测器工具类
  - `SquareDetector` - 正方形模型（使用SSCMA原生后处理）
- `udp_service.h/cpp` - UDP发送服务（独立线程）

## 编译

### 前置条件

在构建此解决方案之前，请确保已按照主项目文档设置 **ReCamera-OS** 环境：

🔗 **[SSCMA Example for SG200X - Main README](../../README.md)**

### 编译步骤

```bash
cd /home/steven/sscma-example-sg200x/solutions/sesg-projet/face_udp
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)
```

生成的可执行文件：`build/face_udp`

## 使用方法

```bash
./face_udp <yolo_face.cvimodel> <age_gender_race.cvimodel> <emotion.cvimodel> [single|multi] [threshold] [skip] [udp_ip] [udp_port] [log_every_n_infer]
```

### 参数说明

- `yolo_face.cvimodel` - YOLO 人脸检测模型（必需）
- `age_gender_race.cvimodel` - 年龄/性别/种族模型（必需）
- `emotion.cvimodel` - 表情模型（必需）
- `single|multi` - YOLO 头类型（可选，默认 multi；传 single 启用单输出模型路径）
- `threshold` - 检测阈值（可选，multi 默认0.5；single 默认0.7）
- `skip` - 每N帧推理1次（可选，multi 默认3；single 默认1）
- `udp_ip` / `udp_port` - UDP目标（可选，只有同时提供才启用 UDP/CH1）
- `log_every_n_infer` - 每N次推理打印一次日志（可选，默认20；传0关闭）

### 使用示例

1. **标准用法**
```bash
./face_udp yolo_face.cvimodel age_gender_race.cvimodel emotion.cvimodel
```

2. **自定义UDP目标**
```bash
./face_udp yolo_face.cvimodel age_gender_race.cvimodel emotion.cvimodel multi 0.5 3 192.168.1.100 5001
```

## 接收端

使用Python客户端接收UDP数据流：

```bash
cd /home/steven/sscma-example-sg200x/solutions/sesg-projet/face_udp
python3 udp_receiver.py
```

## 性能统计

程序会每2秒输出一次详细的性能统计：

```
========== 性能统计 (最近2秒) ==========
观感 FPS: 30.2 | UDP FPS: 9.8
总帧数: 300 | 推理次数: 100

平均推理耗时:
  前处理: 0.9 ms
  推理:   35.5 ms
  后处理: 32.3 ms
  总计:   68.7 ms

平均UDP耗时: 2.1 ms
======================================
```

### 性能指标说明

- **观感 FPS** - 视频帧率（包含所有帧）
- **UDP FPS** - UDP实际发送帧率
- **前处理** - 预处理耗时（数据准备）
- **推理** - TPU推理耗时
- **后处理** - 结果解析耗时
- **UDP耗时** - 帧拷贝+缓冲耗时

## 预期输出

执行应用程序时，将发生以下情况：

1. 加载指定的模型
2. 摄像头开始实时捕获帧
3. 对每一帧进行推理，并显示检测结果
4. 显示捕获、推理和总处理时间的计时统计
5. 检测结果包括找到的对象数量及其边界框和置信度分数

### 示例输出消息

```
Frame 1: Capture: 15 ms, Inference: 25 ms, Total: 48 ms
Detections: 2 objects found
- Class 0: 0.85 confidence at [100,200,150,250]
- Class 1: 0.72 confidence at [300,400,350,450]
```

每10帧：

```
Average processing time per frame: 50.00 ms (over 10 frames)
```

## 重要说明

- 本工程只支持 **正方形输入模型**（例如 640x640）。如果模型输入为非正方形，会直接报错并退出。
- 非正方形/不同分辨率模型的"绕过 SSCMA + 自定义后处理"会放到单独的 demo 工程中。
- 相机分辨率自动设置为匹配模型的输入要求
- 物理地址模式已启用，以实现高效的内存处理

## 技术细节

1. **双缓冲机制** - UDP发送使用双缓冲，避免帧丢失
2. **物理地址输入** - 相机输出直接用于推理，减少拷贝
3. **跳帧推理** - 降低CPU/TPU负载，缓存上次结果用于非推理帧
4. **独立线程** - UDP发送在独立线程，不阻塞主循环
5. **详细计时** - 每个阶段独立计时，便于性能分析

## 参考

有关 SSCMA 框架的更多详细信息，请参阅 [SSCMA-Micro 文档](https://github.com/Seeed-Studio/SSCMA-Micro)。

此示例作为使用 SSCMA 框架和摄像头输入进行实时对象检测的基本介绍。用户可以修改代码并使其适应其特定需求，尝试不同的模型和阈值。
