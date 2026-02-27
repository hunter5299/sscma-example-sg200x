**Groove Features**

- **用途**: 这是一个基于 OpenCV 的纯终端工具，用于从摄像头实时检测并计数“凹槽/竖条”特征。程序在每帧处理完成后在终端打印检测到的 `count` 和该帧处理耗时（ms）。不进行可视化显示。

- **主要处理流程**: 摄像头取帧 -> 裁剪固定 ROI -> 转灰度 -> 固定阈值二值化（反转）-> 竖向形态学开运算 -> 轮廓筛选（面积/宽度/长宽比）-> 计数并打印。

- **默认参数**（在源代码 `main.cpp` 中定义）:
  - ROI: (320,141)-(370,183)
  - 固定阈值: 50
  - 垂直核长度: 5
 这些常量可以直接在 `main.cpp` 中修改并重新编译，后续可按需改成命令行参数。

- **依赖**:
  - 仓库内的交叉编译环境（`SG200X_SDK_PATH` 环境变量用于 CMake 查找 SDK）
  - `sscma-micro`（仓库内已有，作为组件依赖）
  - OpenCV 链接由工程 CMake 通过 `opencv_core` 与 `opencv_imgproc` 等组件引入（根据仓库的构建配置使用 SDK 中的 OpenCV 库）

- **构建（在开发机上运行 CMake 配置并交叉编译）**:

```bash
# 示例：确保 SG200X_SDK_PATH 已指向你的 SDK 根目录（无换行字符）
export SG200X_SDK_PATH=/path/to/sg2002_recamera_emmc
cd /path/to/sscma-example-sg200x
rm -rf build-groove_features
SG200X_SDK_PATH="$SG200X_SDK_PATH" cmake -S solutions/sesg-projet/groove_features -B build-groove_features -DCMAKE_BUILD_TYPE=Release
cmake --build build-groove_features -j
```

- **构建产物**:
  - 可执行文件位于构建目录 `build-groove_features/` 下，名称为 `groove_features`。

- **运行**（在目标设备或可以运行该交叉二进制的环境中）:

```bash
# 直接运行（程序会从默认摄像头采集并在终端打印结果）
./groove_features
```

- **UDP 推流（图像 + 框数据）**:
  - 程序支持通过 SeSg `stream_udp` 发送：**JPEG 图像** + **凹槽框(bbox)列表**。
  - 运行方式与本仓库其他 `sesg-projet` demo 一致：传入 `udp_ip udp_port` 即启用。

```bash
# 在 SG2002 端：启用 UDP 发送到上位机
./groove_features 192.168.2.101 5001
```

  - 发送的数据包含：
    - JPEG（来自配置的 JPEG 通道）
    - det 列表（每个凹槽一个 bbox），bbox 坐标为 **归一化 (0..1)** 的 (cx,cy,w,h)

- **上位机 Python 接收脚本**:

```bash
# PC 端监听并显示（需要 GUI）
python3 solutions/sesg-projet/groove_features/tools/udp_receiver.py --port 5001 --show

# 仅打印 det（无 GUI 也可以）
python3 solutions/sesg-projet/groove_features/tools/udp_receiver.py --port 5001 --print-dets

# 保存接收到的 JPEG（可选叠框保存：--save-overlay）
python3 solutions/sesg-projet/groove_features/tools/udp_receiver.py --port 5001 --save --out ./frames
```

  - Python 依赖（仅在需要显示/叠框/解码时）:

```bash
pip install opencv-python numpy
```

- **日志输出示例**:

```
[groove_features] start (ROI=(320,141)-(370,183), thr=50, klen=5)
[groove_features] count=3 frame_ms=7.42
[groove_features] count=2 frame_ms=6.87
...
```

- **注意事项**:
  - 当前实现直接使用摄像头设备（通过仓库的 `Device/Camera` 接口）。运行时请保证目标板的摄像头已连接且 SDK 环境正确配置。
  - 若需要调整 ROI、阈值或核大小，最快的方法是修改 `solutions/sesg-projet/groove_features/main/main.cpp` 中对应常量并重新编译。
  - UDP 模式下，程序会额外打印 `udp_ms`（一次 sendLatest 的耗时）；若刚启动时 `udp_ms=-1`，通常表示 JPEG 线程尚未拿到最新帧，稍等片刻即可。

- **文件位置**:
  - 主程序: solutions/sesg-projet/groove_features/main/main.cpp
  - CMake: solutions/sesg-projet/groove_features/CMakeLists.txt
  - 组件 CMake: solutions/sesg-projet/groove_features/main/CMakeLists.txt
  - UDP 接收脚本: solutions/sesg-projet/groove_features/tools/udp_receiver.py


