#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Receive SeSg stream_udp packets for groove_features (JPEG + groove bboxes).

Packet layout (little-endian), produced by `udp_service::UDPSender`:
- magic(u32)
- width(u32), height(u32), payload_size(u32), det_count(u32), fmt(u32)
- fid(u32)
- det_bytes (det_count * sizeof(UDPGrooveDet))
- payload bytes (JPEG if fmt==1 else RGB888)

groove_features sends:
struct UDPGrooveDet {
  float bbox_x; // center-x, normalized to full frame (0..1)
  float bbox_y; // center-y, normalized
  float bbox_w; // width, normalized
  float bbox_h; // height, normalized
  float score;
  int32_t id;
};
Size = 5*4 + 4 = 24 bytes

Usage:
  python3 udp_receiver.py --port 5001 --show
  python3 udp_receiver.py --port 5001 --print-dets

Dependencies (optional for --show/--save-overlay):
  pip install opencv-python numpy
"""

import argparse
import os
import socket
import struct
import time
from typing import Optional, Tuple, List

DET_SIZE = 24
DEFAULT_MAGIC = 0x47524F56  # "GROV"

# ROI / processing constants (match device-side defaults)
# 设备端算法 ROI 是按 640x480 写死的，但 UDP 下行 JPEG 可能是 320x240。
# 因此接收端需要把 ROI 按分辨率比例缩放到当前帧。
ROI_X = 425
ROI_Y = 268
ROI_X2 = 520
ROI_Y2 = 355
ROI_REF_W = 640
ROI_REF_H = 480
FIXED_THRESHOLD = 50
FIXED_KERNEL_LEN = 5


def parse_args():
    p = argparse.ArgumentParser(description="Receive groove_features frames (JPEG + groove bboxes)")
    p.add_argument("--ip", default="0.0.0.0", help="listen ip (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=5001, help="listen port (default: 5001)")
    p.add_argument("--magic", type=lambda x: int(x, 0), default=DEFAULT_MAGIC, help="magic u32 (default: 0x47524F56)")
    p.add_argument("--show", action="store_true", help="display window (requires GUI + opencv-python)")
    p.add_argument("--no-show", action="store_true", help="disable display window")
    p.add_argument("--print-dets", action="store_true", help="print bbox list")
    p.add_argument("--save", action="store_true", help="save received JPEG frames (default: off)")
    p.add_argument("--out", default="./frames", help="output dir for --save")
    p.add_argument("--save-overlay", action="store_true", help="when saving, draw bboxes on image (requires opencv)")
    return p.parse_args()


def try_import_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        return cv2, np
    except Exception:
        return None, None


def find_magic(buf: bytearray, magic: int) -> int:
    return buf.find(struct.pack("<I", magic))


def parse_one_frame(buf: bytearray, magic: int) -> Tuple[Optional[dict], int]:
    idx = find_magic(buf, magic)
    if idx < 0:
        if len(buf) > 4 * 1024 * 1024:
            return None, len(buf)
        return None, 0
    if idx > 0:
        return None, idx

    if len(buf) < 28:
        return None, 0

    magic_u32, w, h, payload_size, det_count, fmt, fid = struct.unpack_from("<7I", buf, 0)
    if magic_u32 != magic:
        return None, 4

    det_bytes_len = det_count * DET_SIZE
    total = 28 + det_bytes_len + payload_size
    if len(buf) < total:
        return None, 0

    det_bytes = bytes(buf[28:28 + det_bytes_len])
    payload = bytes(buf[28 + det_bytes_len: total])

    return {
        "magic": magic_u32,
        "w": w,
        "h": h,
        "payload_size": payload_size,
        "det_count": det_count,
        "fmt": fmt,
        "fid": fid,
        "det_bytes": det_bytes,
        "payload": payload,
    }, total


def parse_dets(det_bytes: bytes) -> List[dict]:
    out: List[dict] = []
    if not det_bytes:
        return out
    n = len(det_bytes) // DET_SIZE
    for i in range(n):
        off = i * DET_SIZE
        bbox_x, bbox_y, bbox_w, bbox_h, score, det_id = struct.unpack_from("<5fi", det_bytes, off)
        out.append({
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "score": score,
            "id": det_id,
        })
    return out


def bbox_to_xyxy_norm(d: dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = float(d.get("bbox_x", 0.0)) * img_w
    cy = float(d.get("bbox_y", 0.0)) * img_h
    bw = float(d.get("bbox_w", 0.0)) * img_w
    bh = float(d.get("bbox_h", 0.0)) * img_h

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


def main():
    args = parse_args()

    cv2, np = try_import_cv2()

    # 默认展示（当安装了 opencv 时默认显示），除非显式传入 --no-show。
    # 保留 --show/--no-show 强制开关，但不再依赖 DISPLAY 环境变量。
    if args.no_show:
        enable_show = False
    elif args.show:
        enable_show = True
    else:
        enable_show = (cv2 is not None and np is not None)

    if args.save:
        os.makedirs(args.out, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 50 * 1024 * 1024)
    sock.bind((args.ip, args.port))
    sock.settimeout(1.0)

    print(f"[UDP] listening on {args.ip}:{args.port}, magic=0x{args.magic:08X}")

    buf = bytearray()
    last_fid = None

    while True:
        try:
            data, _addr = sock.recvfrom(65535)
            if data:
                buf.extend(data)

            while True:
                frame, consumed = parse_one_frame(buf, args.magic)
                if consumed > 0:
                    del buf[:consumed]
                if frame is None:
                    break

                fid = frame["fid"]
                if last_fid is not None and fid > last_fid + 1:
                    print(f"[UDP] dropped {fid - last_fid - 1} frames")
                last_fid = fid

                dets = parse_dets(frame["det_bytes"])
                if args.print_dets:
                    print(f"[GROOVE] fid={fid} det_count={len(dets)}")
                    for d in dets:
                        print(
                            f"  id={d['id']} score={d['score']:.2f} "
                            f"cx={d['bbox_x']:.4f} cy={d['bbox_y']:.4f} w={d['bbox_w']:.4f} h={d['bbox_h']:.4f}"
                        )

                img = None
                if (enable_show or args.save_overlay) and cv2 is not None and np is not None:
                    if frame["fmt"] == 1:
                        img = cv2.imdecode(np.frombuffer(frame["payload"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    else:
                        arr = np.frombuffer(frame["payload"], dtype=np.uint8)
                        img = arr.reshape((frame["h"], frame["w"], 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if img is not None:
                        ih, iw = img.shape[:2]

                        # 计算二值化+形态学处理，填充到全尺寸用于显示（与 Python 参考实现一致）
                        morph_full = None
                        try:
                            # 将 640x480 参考 ROI 缩放到当前图像尺寸
                            sx = iw / float(ROI_REF_W)
                            sy = ih / float(ROI_REF_H)
                            rx1 = int(round(ROI_X * sx))
                            ry1 = int(round(ROI_Y * sy))
                            rx2 = int(round(ROI_X2 * sx))
                            ry2 = int(round(ROI_Y2 * sy))

                            x1 = max(0, min(iw, rx1))
                            y1 = max(0, min(ih, ry1))
                            x2 = max(0, min(iw, rx2))
                            y2 = max(0, min(ih, ry2))

                            morph_full = np.zeros(img.shape, dtype=np.uint8)
                            if x2 > x1 and y2 > y1:
                                roi = img[y1:y2, x1:x2]
                                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                _, binary = cv2.threshold(gray, FIXED_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
                                k_len = max(1, FIXED_KERNEL_LEN)
                                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
                                morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
                                morph = cv2.bitwise_not(morph)

                                morph_rgb = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
                                morph_full[y1:y2, x1:x2] = morph_rgb

                            # ROI 框（黄色）
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.rectangle(morph_full, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        except Exception:
                            morph_full = np.zeros(img.shape, dtype=np.uint8)

                        # 在主图和二值图上绘制框与编号
                        for d in dets:
                            x1b, y1b, x2b, y2b = bbox_to_xyxy_norm(d, iw, ih)
                            cv2.rectangle(img, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)
                            cv2.putText(img, f"#{d['id']}", (x1b, max(0, y1b - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            if morph_full is not None:
                                cv2.rectangle(morph_full, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)

                        cv2.putText(img, f"fid={fid} count={len(dets)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        # 显示二值图窗口（即使 ROI 失败也显示黑底，避免只有一个窗口）
                        if enable_show:
                            cv2.imshow("Processed", morph_full)

                if args.save:
                    out_path = os.path.join(args.out, f"frame_{fid}.jpg")
                    if args.save_overlay and img is not None and cv2 is not None:
                        cv2.imwrite(out_path, img)
                    else:
                        # raw JPEG bytes
                        with open(out_path, "wb") as f:
                            f.write(frame["payload"])

                if enable_show and img is not None and cv2 is not None:
                    cv2.imshow("groove_features", img)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        raise KeyboardInterrupt

        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERR] {e}")
            time.sleep(0.05)

    sock.close()
    if cv2 is not None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
