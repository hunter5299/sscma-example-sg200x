# Depth Anything NPU

## 概述

`depth_anything_npu` 是一个 reCamera SG200X NPU 深度估计 demo，使用 Depth Anything V2 ViT-S 转换得到的 CVIMODEL。程序支持两种模式：

- 静态图片：输出灰度深度图、彩色深度图、原图/深度并排图。
- 摄像头：从 reCamera JPEG 通道取帧，输出连续的原图/深度并排 PNG，可再用 `ffmpeg` 合成视频。

当前推荐模型是 `224x224 INT8`，能在 reCamera 上运行；`224x224 BF16` 曾因 ION 内存不足失败，因此没有放入本 demo 目录。

## 文件结构

```text
depth_anything_npu/
  CMakeLists.txt
  README.md
  main/
    CMakeLists.txt
    main.cpp
  models/
    depth_anything_v2_vits_224_int8_min8.cvimodel
    depth_anything_v2_vits_168_bf16.cvimodel
```

## 模型

已放入本 demo 的模型：

- `models/depth_anything_v2_vits_224_int8_min8.cvimodel`
  - 推荐使用
  - 输入 `[1,224,224,3]`，运行时 U8 RGB packed
  - 输出 `[1,1,224,224]`
  - reCamera 摄像头测试平均 NPU 推理约 `1693 ms/frame`

- `models/depth_anything_v2_vits_168_bf16.cvimodel`
  - 较小 BF16 可行性模型
  - 输入 `[1,168,168,3]`，运行时 U8 RGB packed
  - 输出 `[1,1,168,168]`
  - 单图 NPU 推理约 `656 ms`

模型转换产物建议放在源码树外，例如：

```text
$WORK_DIR/recamera_depth_anything
```

## 编译

在开发主机上编译：

```bash
export REPO_ROOT=<path-to-sscma-example-sg200x>
export SDK_ROOT=<path-to-sg200x-sdk>
export TOOLCHAIN_BIN=<path-to-riscv64-toolchain>/bin

export SG200X_SDK_PATH="$SDK_ROOT"
export PATH="$TOOLCHAIN_BIN:$PATH"
cd "$REPO_ROOT/solutions/sesg-project/depth_anything_npu"
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build -j$(nproc)
```

生成文件：

```text
build/depth_anything_npu
```

## 部署到 reCamera

复制二进制和模型到设备：

```bash
export RECAMERA_HOST=<recamera-ip>
export DEMO_DIR=<recamera-demo-dir>

ssh "recamera@$RECAMERA_HOST" "mkdir -p '$DEMO_DIR/models'"
scp build/depth_anything_npu "recamera@$RECAMERA_HOST:$DEMO_DIR/"
scp models/*.cvimodel "recamera@$RECAMERA_HOST:$DEMO_DIR/models/"
```

运行前建议停止会占用摄像头/NPU 的服务：

```bash
for svc in /etc/init.d/S*node-red* /etc/init.d/S*sscma-node* /etc/init.d/S*sscma-supervisor*; do
  [ -x "$svc" ] && "$svc" stop 2>/dev/null || true
done
killall -q depth_anything_npu 2>/dev/null || true
```

运行时需要 root 权限和库路径：

```bash
export RECAMERA_LIB_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/usr/lib
```

## 静态图片运行

命令格式：

```bash
./depth_anything_npu model.cvimodel input_image output_prefix [repeat]
```

示例：

```bash
cd "$DEMO_DIR"
sudo env LD_LIBRARY_PATH="$RECAMERA_LIB_PATH" \
  ./depth_anything_npu \
    ./models/depth_anything_v2_vits_224_int8_min8.cvimodel \
    ./teaser.png \
    ./depth_teaser_224 \
    1
```

输出：

```text
depth_teaser_224_gray.png
depth_teaser_224_color.png
depth_teaser_224_side_by_side.png
```

## 摄像头运行

命令格式：

```bash
./depth_anything_npu --camera model.cvimodel output_dir [frames] [skip]
```

示例：

```bash
cd "$DEMO_DIR"
mkdir -p live_224_int8_jpeg
sudo env LD_LIBRARY_PATH="$RECAMERA_LIB_PATH" \
  ./depth_anything_npu \
    --camera ./models/depth_anything_v2_vits_224_int8_min8.cvimodel \
    ./live_224_int8_jpeg \
    12 \
    1
```

程序会保存并排 PNG：

```text
live_224_int8_jpeg/frame_0000.png
live_224_int8_jpeg/frame_0001.png
...
```

合成视频：

```bash
ffmpeg -y -framerate 2 -i live_224_int8_jpeg/frame_%04d.png \
  -pix_fmt yuv420p recamera_live_depth_anything_224_int8_jpeg.mp4
```

## 预期日志

`224 INT8` 摄像头模式示例：

```text
input[0]: shape=[1,224,224,3], type=U8, bytes=150528
saved=0, run_ms=..., depth_min=..., depth_max=..., path=...
saved_frames=12, avg_npu_run_ms=...
```

`168 BF16` 单图模式示例：

```text
input[0]: shape=[1,168,168,3], type=U8, bytes=84672
run[0]_ms=656.043, ret=0
output[0]: shape=[1,1,168,168], type=F32, bytes=112896
```

## 注意事项

- 摄像头模式使用 JPEG 通道并通过 OpenCV 解码，再 resize 到模型输入尺寸。之前直接把 RGB888 VPSS buffer 当 OpenCV packed RGB 使用时，视觉证据不稳定。
- `224x224 BF16` 模型虽然可以编译，但在 reCamera 上出现 ION 内存不足，当前不建议使用。
- 该 demo 输出的是相对深度可视化，不是带真实单位的 metric depth。
- 测试结束后恢复服务：

```bash
/etc/init.d/S03node-red start
/etc/init.d/S91sscma-node start
/etc/init.d/S93sscma-supervisor start
```

## 参考证据

历史测试报告可放在项目文档目录或发布说明中，例如：

```text
docs/DEPLOY_REPORT.md
```

历史视频证据建议使用相对路径或公开链接，例如：

```text
docs/evidence/recamera_live_depth_anything_224_int8_jpeg.mp4
```
