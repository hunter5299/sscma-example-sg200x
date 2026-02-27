#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTSP 接收端 (PC端运行)

face_rtsp 推流为 H264/RTSP，本脚本用于在 PC 上接收 rtsp://<device-ip>:<port>/<session> 并低延时显示。

依赖：pip install opencv-python

示例：
  python3 rtsp_receiver.py --host 192.168.1.10 --port 8554 --session live
  python3 rtsp_receiver.py --url rtsp://192.168.1.10:8554/live
"""

import argparse
import os
import time
import cv2


def build_url(host: str, port: int, session: str) -> str:
    session = (session or "").lstrip("/")
    return f"rtsp://{host}:{port}/{session}"


def open_capture(url: str) -> cv2.VideoCapture:
    # 使用 FFMPEG 后端通常更稳
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # 尽量降低缓冲带来的延时（不保证所有 backend 都生效）
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description="RTSP receiver for face_rtsp")
    parser.add_argument("--url", default="", help="完整 RTSP URL，例如 rtsp://192.168.1.10:8554/live")
    parser.add_argument("--host", default="192.168.2.10", help="设备 IP")
    parser.add_argument("--port", type=int, default=8554, help="RTSP 端口")
    parser.add_argument("--session", default="live", help="RTSP session/path")
    parser.add_argument(
        "--transport",
        choices=["tcp", "udp"],
        default="tcp",
        help="RTSP 传输方式（默认 tcp 更抗丢包、PPS/SPS 更稳定）",
    )
    parser.add_argument("--stimeout_ms", type=int, default=5000, help="FFmpeg stimeout (ms)，默认 5000")
    parser.add_argument("--scale", type=float, default=1.0, help="显示缩放倍数")
    parser.add_argument("--reconnect", type=float, default=1.0, help="断开后重连间隔(秒)")
    args = parser.parse_args()

    url = args.url.strip() or build_url(args.host, args.port, args.session)
    print(f"[RTSP] URL: {url}")
    print(f"[RTSP] transport: {args.transport} (OpenCV FFMPEG)")
    print("按 'q' 退出")

    # OpenCV 的 FFmpeg 后端可以通过环境变量强制 RTSP/TCP。
    # 这对减少丢包引发的 H264 解码报错（missing PPS / no frame）很关键。
    # 注：仅对 CAP_FFMPEG 生效。
    stimeout_us = max(0, int(args.stimeout_ms)) * 1000
    if args.transport == "tcp":
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;tcp|stimeout;{stimeout_us}|max_delay;0",
        )
    else:
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;udp|stimeout;{stimeout_us}|max_delay;0",
        )

    cap = open_capture(url)
    last_ok = time.time()

    try:
        while True:
            if not cap.isOpened():
                print("[RTSP] 打开失败，等待重连...")
                time.sleep(max(0.1, args.reconnect))
                cap.release()
                cap = open_capture(url)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                # 读不到帧：短暂等待并尝试重连
                if time.time() - last_ok > 2.0:
                    print("[RTSP] 读帧失败，重连...")
                    cap.release()
                time.sleep(0.01)
                continue

            last_ok = time.time()

            if args.scale != 1.0:
                frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)

            cv2.imshow("RTSP Receiver", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
