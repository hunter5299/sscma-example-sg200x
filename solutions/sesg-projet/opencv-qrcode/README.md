# opencv-qrcode

这是一个用于“对比 OpenCV 自带二维码识别效果”的独立工程：
- 代码结构与 `solutions/sesg-projet/qrcode-udp` 保持一致
- 仅把二维码解码从 ZXing 替换为 OpenCV `cv::QRCodeDetector`

## 重要前提（你现在的环境还不满足）

要使用 `cv::QRCodeDetector`，你的 OpenCV 必须包含 `objdetect` 模块：
- 头文件：`opencv2/objdetect.hpp`
- 库：`libopencv_objdetect.so`（或 `.a`）

根据我刚刚在你的路径里检查到的结果：
- `cvitek_tpu_sdk` 和 `sysroot` 里目前都没有 `opencv2/objdetect.hpp`

所以：这个工程已经创建好，但在你补齐 OpenCV 的 `objdetect` 之前，**它无法编译/链接**。

## 构建

```bash
cd solutions/sesg-projet/opencv-qrcode
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
```

## 运行

与 `qrcode-udp` 一样（位置参数）：

```bash
./opencv_qrcode                     # 视频流推理（无UDP）
./opencv_qrcode 192.168.2.101 5001  # 视频流推理 + UDP
./opencv_qrcode image /tmp/qr.jpg   # 本地图片推理
```
