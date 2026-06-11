#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTMP 接收端 (PC 端运行)

从 RTMP 服务器(如 SRS)拉取 reCamera 推送的带 YOLO 检测框的 H.264 流并显示/录制。
也可直接用 ffplay/ffmpeg/VLC 播放：
    ffplay rtmp://<srs-ip>:1935/live/recamera
    ffplay http://<srs-ip>:8080/live/recamera.flv   # HTTP-FLV

依赖：pip install opencv-python

示例：
    python3 rtmp_receiver.py --url rtmp://192.168.2.113:1935/live/recamera
    python3 rtmp_receiver.py --url http://192.168.2.113:8080/live/recamera.flv --record out.mp4 --seconds 15
"""

import argparse
import os
import time
import cv2


def main() -> int:
    ap = argparse.ArgumentParser(description="RTMP/HTTP-FLV receiver for rtmp_yolo")
    ap.add_argument("--url", default="rtmp://192.168.2.113:1935/live/recamera",
                    help="RTMP 或 HTTP-FLV 地址")
    ap.add_argument("--record", default="", help="录制到 mp4 文件")
    ap.add_argument("--seconds", type=int, default=0, help="录制/播放秒数(0=直到 q)")
    args = ap.parse_args()

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|max_delay;0")
    print(f"[rtmp] open {args.url}")
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    writer = None
    t0 = time.time()
    last_ok = time.time()
    try:
        while True:
            if not cap.isOpened():
                print("[rtmp] open failed, retry...")
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.time() - last_ok > 5.0:
                    print("[rtmp] read timeout, exit")
                    break
                time.sleep(0.01)
                continue
            last_ok = time.time()

            if args.record and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                print(f"[rtmp] recording -> {args.record} ({w}x{h})")
            if writer is not None:
                writer.write(frame)

            cv2.imshow("RTMP + YOLO (reCamera)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.seconds and time.time() - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
