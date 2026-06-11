# RTMP + YOLO AI IPC 部署报告

- **Demo 名称**：rtmp_yolo
- **日期**：2026-06-11
- **设备**：reCamera（SG2002 / cv181x，RISC-V，reCamera OS 0.2.4）
- **目标**：reCamera 摄像头经 YOLO11n 检测、设备端硬件画框后，把带框 H.264 通过 RTMP 推到流媒体服务器；seeed 上用 SRS 接收并拉流验收。

## 1. 结论

端到端验收通过：

- reCamera 检测引擎 1280x720 @ ~26.7 FPS，YOLO11n 推理 ~38 ms/帧
- 检测框 + 类别/置信度标签经 RGN/OSD 硬件叠加烧进 H.264
- 设备 ffmpeg 把本地 RTSP `-c copy` 转封装推到 SRS RTMP（无重编码）
- SRS 收到 `/live/recamera`（H264 Baseline 1280x720），seeed ffmpeg 拉流录制成功
- 录制视频确认含 person 检测框（夜间暗光场景，17/70 抽样帧含框）

## 2. 架构

```
reCamera (rtmp_yolo + ffmpeg relay)
  相机 CH0 RGB888 → YOLO11n(NPU) → 检测结果
  相机 CH2 H264  ← RGN/OSD 硬件画框 ← 检测结果
       ↓ 本地 RTSP 127.0.0.1:8554/live
  ffmpeg -rtsp_transport tcp -i rtsp://127.0.0.1:8554/live -c copy -f flv rtmp://<srs>:1935/live/recamera
       ↓ RTMP
seeed: SRS 5 (docker, host net) :1935 RTMP / :8080 HTTP-FLV / :1985 API
       ↓ 拉流
seeed ffmpeg / PC ffplay / rtmp_receiver.py
```

## 3. 复用与新增

| 模块 | 来源 |
|------|------|
| 相机 + VPSS + VENC + RGN 画框 + 本地 RTSP | 复用 onvif_yolo 的引擎（`SesgRtspVpssStreamer` + 修正后的 stream_rtsp RGN） |
| YOLO 检测 + 文字标签 | 复用 onvif_yolo（detector_utils + font5x7 + coco_labels） |
| RTMP 推流 | **新增**：设备 ffmpeg RTSP→RTMP relay，`run_rtmp.sh` 一键封装 |
| PC 验收 | **新增** `rtmp_receiver.py` |

`rtmp_yolo` 程序本体 = onvif_yolo 去掉 ONVIF 服务，保留检测 + 本地 RTSP（CH2），由 `run_rtmp.sh` 拉起 ffmpeg relay。

## 4. 关键决策与踩坑

1. **在进程内用 libavformat 直接推 RTMP 失败**
   - 现象 A（LD_LIBRARY_PATH 用 /mnt/system/lib）：`avformat_alloc_output_context2` 找不到 FLV muxer（该 .so 仅 2 个 muxer，裁剪版）。
   - 现象 B（改用 /usr/lib 的 libavformat，6 muxer）：FLV muxer 有了、能连上 SRS，但 SRS 报 `demux SPS/PPS: avc decode sequence header` 失败——因为没有把 SPS/PPS 以 AVCC extradata 提供给 muxer，且发的是 Annex-B 而非 AVCC。
   - 结论：设备运行时 ffmpeg 库为裁剪版，进程内推 RTMP 需要额外做 Annex-B→AVCC + extradata 提取，复杂且脆弱。
   - **决策**：改用设备自带的 `ffmpeg` 可执行文件（muxer 完整）做 RTSP→RTMP `-c copy` 转封装。这是工业 IPC 接入流媒体/CDN 的常见做法，鲁棒、零重编码。

2. **进程内 fork() ffmpeg 不稳定**
   - 在多线程程序里 fork()+execlp 子进程出现 `[ffmpeg]` defunct，relay 不工作（fork-after-threads 不安全）。
   - **决策**：不在 C 程序里 fork，改由 `run_rtmp.sh` shell 脚本分别拉起检测引擎和 ffmpeg relay，relay 带断线重连循环。

3. **夜间暗光**：验收时为夜间，画面很暗（平均亮度 ~6/255），检测以 person 为主且间歇。证据帧/GIF 做了 `eq` 提亮以便看清检测框。

## 5. 运维注意

- 退出用 `Ctrl-C` / `kill -TERM`，`run_rtmp.sh` 会清理 ffmpeg 和检测引擎。
- 不要 `kill -9` 检测引擎（残留 VPSS，需 reboot）。
- 运行前停 `S03node-red` / `S91sscma-node` / `S93sscma-supervisor`。
- SRS 启动：`docker run -d --name srs --network host ossrs/srs:5 ./objs/srs -c conf/srs.conf`。

## 6. 真实环境路径（仅内部）

- seeed 仓库：`/home/steven/sscma-example-sg200x/solutions/sesg-project/rtmp_yolo`
- 工具链：`/home/seeed/zsz/TOOL/riscv64-linux-musl-x86_64/bin`
- SDK：`/home/seeed/桌面/sg2002_recamera_emmc`
- 设备运行目录：`/home/recamera/rtmp_yolo`
- SRS：seeed docker 容器 `srs-gb`，host 网络，192.168.2.113

## 7. 云端资产

- Google Drive 根目录：https://drive.google.com/drive/folders/1GOQUMCel7fapbJCWzEEynDIvIt-6Wf5p?usp=drive_link
- 模型：`/reCamera_Shared/Wiki/rtmp_yolo/model/yolo11n_detection_cv181x_int8.cvimodel`
- 证据图片：`/reCamera_Shared/Wiki/rtmp_yolo/evidence/image/`（demo.gif, frame_detection_01.png, frame_detection_bright.png, rtmp_acceptance.txt, rtmp_srs_api.txt）
- 证据视频：`/reCamera_Shared/Wiki/rtmp_yolo/evidence/video/rtmp_yolo_demo.mp4`
