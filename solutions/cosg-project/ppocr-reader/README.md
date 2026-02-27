# ppocr-reader

PP-OCRv3 文字检测与识别应用，运行在 reCamera (Sophgo SG2002/cv181x) 上，利用 TPU 加速推理。

## 功能

- **文字检测**: 基于 DBNet 的场景文字检测，输出四边形文字区域
- **文字识别**: 基于 SVTR-LCNet 的中英文混合识别，支持 6623 个字符（简体中文、英文、数字、标点）
- **RTSP 视频流**: H.264 编码的实时视频流，可通过 VLC 等播放器查看
- **MQTT 输出**: 每帧检测结果以 JSON 格式发布到 MQTT broker
- **两阶段流水线**: 先检测文字区域，再逐区域识别文字内容，结果按从上到下、从左到右排序

## 性能指标

| 指标 | 值 |
|------|-----|
| 检测推理 | ~65 ms |
| 检测模型大小 | 877 KB (INT8) |
| 识别模型大小 | 2.9 MB (INT8) |
| 检测模型 ION 内存 | 6.75 MB |
| 识别模型 ION 内存 | 4.91 MB |
| 摄像头分辨率 | 640x480 |
| RTSP 输出分辨率 | 1280x720 @ 15fps |

## 安装

### 1. 编译

在 x86_64 主机上使用 Docker 交叉编译：

```bash
docker exec ubuntu_dev_x86 bash -c "
  export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc
  export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:\$PATH
  cd /workspace/sscma-example-sg200x/solutions/ppocr-reader
  rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release . && cmake --build build -j4
  cd build && cpack
"
```

产物: `build/ppocr-reader_0.1.0_riscv64.deb`

### 2. 模型转换

模型转换在 `sophgo/tpuc_dev:v3.1` Docker 容器中进行，详见 `model_conversion/recamera_ppocr/`。

```bash
# 启动容器
docker start ppocr_convert

# ONNX 导出 (在 macOS/Linux 主机上)
cd model_conversion/recamera_ppocr
uv run python scripts/export_ppocr.py

# INT8 模型转换 (在 Docker 容器内)
docker exec ppocr_convert bash -c "
  cd /workspace/model_conversion/recamera_ppocr
  bash scripts/convert_det.sh --quantize INT8
  bash scripts/convert_rec.sh --quantize INT8
"
```

产物:
- `ppocr_det_cv181x_int8.cvimodel` (877 KB)
- `ppocr_rec_cv181x_int8.cvimodel` (2.9 MB)

### 3. 部署到 reCamera

```bash
# 复制文件到设备
scp model_conversion/recamera_ppocr/ppocr_det_cv181x_int8.cvimodel recamera@192.168.42.1:/tmp/
scp model_conversion/recamera_ppocr/ppocr_rec_cv181x_int8.cvimodel recamera@192.168.42.1:/tmp/
scp model_conversion/recamera_ppocr/ppocr_keys_v1.txt recamera@192.168.42.1:/tmp/
scp sscma-example-sg200x/solutions/ppocr-reader/build/ppocr-reader_0.1.0_riscv64.deb recamera@192.168.42.1:/tmp/

# SSH 登录设备安装
ssh recamera@192.168.42.1

# 安装 deb 包
sudo opkg install /tmp/ppocr-reader_0.1.0_riscv64.deb

# 复制模型和字典
sudo cp /tmp/ppocr_det_cv181x_int8.cvimodel /userdata/local/models/
sudo cp /tmp/ppocr_rec_cv181x_int8.cvimodel /userdata/local/models/
sudo cp /tmp/ppocr_keys_v1.txt /userdata/local/dict/
```

## 使用

### 启动/停止服务

```bash
# 启动 (自动停止冲突的 node-red、sscma-node 等服务)
sudo /etc/init.d/S92ppocr-reader start

# 停止
sudo /etc/init.d/S92ppocr-reader stop

# 重启
sudo /etc/init.d/S92ppocr-reader restart

# 查看状态
sudo /etc/init.d/S92ppocr-reader status
```

### 查看 RTSP 视频流

```
rtsp://192.168.42.1:8554/live0
```

用 VLC 或 ffplay 打开：
```bash
ffplay rtsp://192.168.42.1:8554/live0
```

### 订阅 MQTT 结果

```bash
mosquitto_sub -h 192.168.42.1 -t "recamera/ppocr/texts" -v
```

### 命令行参数

```
ppocr-reader [options]

Options:
  --det-model PATH     检测模型路径 (默认: /userdata/local/models/ppocr_det_cv181x_int8.cvimodel)
  --rec-model PATH     识别模型路径 (默认: /userdata/local/models/ppocr_rec_cv181x_int8.cvimodel)
  --dict PATH          字典文件路径 (默认: /userdata/local/dict/ppocr_keys_v1.txt)
  --mqtt-host HOST     MQTT broker 地址 (默认: localhost)
  --mqtt-port PORT     MQTT broker 端口 (默认: 1883)
  --mqtt-topic TOPIC   MQTT 发布主题 (默认: recamera/ppocr/texts)
  --no-rtsp            禁用 RTSP 视频流
  --no-mqtt            禁用 MQTT 发布
  -v, --verbose        启用详细日志
  -h, --help           显示帮助信息
```

### 自定义配置

创建 `/etc/ppocr-reader.conf` 可覆盖 init 脚本中的默认参数：

```bash
# /etc/ppocr-reader.conf
DAEMON_OPTS="--det-model /userdata/local/models/ppocr_det_cv181x_int8.cvimodel \
  --rec-model /userdata/local/models/ppocr_rec_cv181x_int8.cvimodel \
  --dict /userdata/local/dict/ppocr_keys_v1.txt \
  --mqtt-topic my/custom/topic \
  --verbose"
```

## MQTT 输出格式

每帧发布一条 JSON 消息到 `recamera/ppocr/texts`：

```json
{
  "timestamp": 1768969602957,
  "frame_id": 42,
  "inference_ms": {
    "detection": 65.2,
    "recognition": 48.3,
    "total": 113.5
  },
  "texts": [
    {
      "box": [[10,20], [200,20], [200,50], [10,50]],
      "text": "Hello World",
      "confidence": 0.95
    },
    {
      "box": [[10,60], [150,60], [150,100], [10,100]],
      "text": "你好世界",
      "confidence": 0.88
    }
  ]
}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | int | Unix 时间戳 (毫秒) |
| `frame_id` | int | 帧序号 |
| `inference_ms.detection` | float | 检测推理耗时 (ms) |
| `inference_ms.recognition` | float | 识别推理耗时 (ms, 所有文字区域累计) |
| `inference_ms.total` | float | 总耗时 (ms) |
| `texts[].box` | array | 文字区域四边形坐标 (顺时针, 左上起) |
| `texts[].text` | string | 识别出的文字内容 |
| `texts[].confidence` | float | 识别置信度 (0-1) |

当画面中没有文字时，`texts` 为空数组 `[]`。

## 模型信息

### 检测模型 (PP-OCRv3 det)

- 架构: MobileNetV3 + RSE-FPN + DBNet head
- 输入: uint8 RGB [1, 480, 480, 3] (预处理已融合)
- 输出: fp32 sigmoid 概率图 [1, 1, 480, 480]
- 后处理: 二值化 → 轮廓提取 → 最小外接矩形 → unclip 扩展

### 识别模型 (PP-OCRv3 rec)

- 架构: SVTR-LCNet
- 输入: uint8 RGB [1, 48, 320, 3] (预处理已融合)
- 输出: fp32 softmax [1, 40, 6625] (40 时间步 x 6625 字符类别)
- 后处理: CTC 贪心解码 (argmax → 去重 → 去 blank → 查字典)
- 字典: ppocr_keys_v1.txt (6623 字符 + space + CTC blank = 6625)

### 量化校准

INT8 量化使用真实场景数据校准：
- 检测: 187 张 ChineseOCRBench 场景文字图片
- 识别: 250 张从检测图片中裁剪的文字区域

关键层 (sigmoid/softmax/attention) 自动使用 bf16 保持精度，其余层为 int8。

## 文件结构

```
ppocr-reader/
├── CMakeLists.txt
├── README.md
├── main/
│   ├── CMakeLists.txt
│   ├── main.cpp               # 入口: 参数解析、相机、RTSP、MQTT、主循环
│   ├── ocr_pipeline.h/cpp     # 两阶段 OCR 流水线
│   ├── text_detector.h/cpp    # DBNet 文字检测
│   ├── text_recognizer.h/cpp  # SVTR 文字识别 + CTC 解码
│   ├── mqtt_publisher.h/cpp   # MQTT JSON 发布
│   └── rtsp_demo.h/c          # RTSP 流
├── control/
│   ├── postinst               # 安装后脚本
│   └── prerm                  # 卸载前脚本
└── rootfs/
    └── etc/init.d/S92ppocr-reader  # SysVinit 服务脚本
```

## 日志

运行日志输出到 `/var/log/ppocr-reader.log`：

```bash
sudo tail -f /var/log/ppocr-reader.log
```

## 恢复设备

测试完成后恢复 reCamera 默认服务：

```bash
sudo /etc/init.d/S92ppocr-reader stop
sudo /etc/init.d/S03node-red start
sudo /etc/init.d/S91sscma-node start
```

卸载 ppocr-reader：

```bash
sudo opkg remove ppocr-reader
```

## 注意事项

- reCamera 摄像头为独占资源，ppocr-reader 启动时会自动停止 node-red、sscma-node 等冲突服务
- 如果出现 VPSS 错误，需要重启设备: `sudo reboot`
- F16 量化在 cv181x 上不支持此模型（tpu_mlir 限制），仅支持 INT8
- 设备默认 SSH 通过 USB NCM 连接: `ssh recamera@192.168.42.1`
