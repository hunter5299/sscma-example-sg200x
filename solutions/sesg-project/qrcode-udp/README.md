# qrcode-udp

在 SG200X / ReCamera 平台上打开摄像头实时识别二维码，并通过 `components/SeSg/stream_udp` 把“画面(JPEG) + 识别结果”通过 UDP 推送到上位机（默认 `192.168.2.10:5000`）。

## 构建

在仓库根目录已配置好工具链/SDK 的前提下：

```bash
cd solutions/sesg-projet/qrcode-udp
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
```

打包（可选）：

```bash
cd build && cpack
```

## 设备端运行

默认推送到 `192.168.2.10:5000`，JPEG 使用 CH1。

```bash
qrcode_udp
```

常用参数：

- `--udp-ip 192.168.2.10`：上位机 IP
- `--udp-port 5000`：上位机端口
- `--cam-w 640 --cam-h 480`：算法取帧分辨率（RGB888）
- `--jpeg-w 320 --jpeg-h 240 --jpeg-fps 10 --jpeg-ch 1`：UDP 推送 JPEG 通道参数

说明：在本仓库的 `CameraSG200X` 实现里，`retrieveFrame(MA_PIXEL_FORMAT_JPEG)` 固定读取 CH1，
因此即使把 `--jpeg-ch` 设为其它值也不会生效（程序会自动回落到 CH1）。

## 上位机接收显示

脚本在 `tools/udp_receiver.py`：

```bash
python3 tools/udp_receiver.py --port 5000 --show --print-dets
```

依赖：`pip install opencv-python numpy`

## UDP 结果结构

每帧会附带 0 或 1 个识别结果：

```c
struct UDPQrCodeResult {
  float bbox_x;   // 目前固定为 0
  float bbox_y;   // 目前固定为 0
  float bbox_w;   // 目前固定为 0
  float bbox_h;   // 目前固定为 0
  float score;    // 识别成功为 1.0
  char  text[128];
};
```

画面采用 `sesg::stream_udp::SesgJpegUdpStreamer` 发送的 JPEG 数据。
