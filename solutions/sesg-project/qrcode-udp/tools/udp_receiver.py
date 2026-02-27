#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Receive SeSg stream_udp packets for qrcode-udp (JPEG + QR result).

Packet layout (little-endian), produced by `udp_service::UDPSender`:
- magic(u32)
- width(u32), height(u32), payload_size(u32), det_count(u32), fmt(u32)
- fid(u32)
- det_bytes (det_count * sizeof(UDPQrCodeResult))
- payload bytes (JPEG if fmt==1 else RGB888)

qrcode-udp uses:
struct UDPQrCodeResult {
  float bbox_x;
  float bbox_y;
  float bbox_w;
  float bbox_h;
  float score;
  char  text[128];
};
Size = 5*4 + 128 = 148 bytes

Usage:
  python3 udp_receiver.py --port 5001 --show --print-dets

Dependencies:
  pip install opencv-python numpy
"""

import argparse
import socket
import struct
import time
from typing import Optional, Tuple, List

QR_DET_SIZE = 148
DEFAULT_MAGIC = 0x51524344  # "QRCD"


def parse_args():
    p = argparse.ArgumentParser(description="Receive qrcode-udp frames (JPEG + QR results)")
    p.add_argument("--ip", default="0.0.0.0", help="listen ip (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=5001, help="listen port (default: 5001)")
    p.add_argument("--magic", type=lambda x: int(x, 0), default=DEFAULT_MAGIC, help="magic u32 (default: 0x51524344)")
    p.add_argument("--show", action="store_true", help="display frames (requires opencv-python)")
    p.add_argument("--print-dets", action="store_true", help="print decoded QR text")
    return p.parse_args()


def try_import_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        return cv2, np
    except Exception:
        return None, None


def find_magic(buf: bytearray, magic: int) -> int:
    # scan for little-endian u32 == magic
    m = struct.pack("<I", magic)
    return buf.find(m)


def parse_one_frame(buf: bytearray, magic: int) -> Tuple[Optional[dict], int]:
    """Try parse one frame from buf. Return (frame_dict_or_None, consumed_bytes)."""
    idx = find_magic(buf, magic)
    if idx < 0:
        # if buffer grows too much without magic, drop it
        if len(buf) > 4 * 1024 * 1024:
            return None, len(buf)
        return None, 0

    if idx > 0:
        return None, idx

    # Need at least 28 bytes for header (magic + 5*4 + fid)
    if len(buf) < 28:
        return None, 0

    magic_u32, w, h, payload_size, det_count, fmt, fid = struct.unpack_from("<7I", buf, 0)
    if magic_u32 != magic:
        return None, 4  # should not happen, but move forward

    det_bytes_len = det_count * QR_DET_SIZE
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
    n = len(det_bytes) // QR_DET_SIZE
    for i in range(n):
        off = i * QR_DET_SIZE
        bbox_x, bbox_y, bbox_w, bbox_h, score = struct.unpack_from("<5f", det_bytes, off)
        text_bytes = det_bytes[off + 5 * 4: off + QR_DET_SIZE]
        text = text_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        out.append({
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "score": score,
            "text": text,
        })
    return out


def main():
    args = parse_args()
    cv2, np = try_import_cv2() if args.show else (None, None)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 50 * 1024 * 1024)
    sock.bind((args.ip, args.port))
    sock.settimeout(1.0)

    print(f"[UDP] listening on {args.ip}:{args.port}, magic=0x{args.magic:08X}")

    buf = bytearray()
    last_fid = None
    last_text = None
    first_pkt = True

    while True:
        try:
            data, _addr = sock.recvfrom(65535)
            if data:
                if first_pkt:
                    print(f"[UDP] first packet: {len(data)} bytes")
                    first_pkt = False
                buf.extend(data)

            # Try parse as many frames as possible
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
                if args.print_dets and dets:
                    t = dets[0]["text"]
                    if t != last_text:
                        print(f"[QR] {t}")
                        last_text = t

                if cv2 is not None and np is not None:
                    if frame["fmt"] == 1:
                        img = cv2.imdecode(np.frombuffer(frame["payload"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    else:
                        # RGB888
                        arr = np.frombuffer(frame["payload"], dtype=np.uint8)
                        img = arr.reshape((frame["h"], frame["w"], 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if img is not None:
                        # overlay
                        if dets:
                            text = dets[0]["text"]
                            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(img, f"fid={fid}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("qrcode-udp", img)
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
