#!/usr/bin/env python3
import socket
import struct
import sys
import os
try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

# Simple UDP receiver for the protocol used by udp_service.
# Stream is fragmented into UDP packets (max payload ~60KB) and must be reassembled.
# Header (little-endian):
# magic u32
# width u32
# height u32
# payload_size u32
# det_count u32
# fmt u32
# fid u32
# det_bytes (det_count * sizeof(T))
# payload bytes (JPEG)

MAGIC = 0x55445044
DET_STRUCT = '<fffffi'  # x,y,w,h,score (float), target (int)
DET_SIZE = struct.calcsize(DET_STRUCT)

if len(sys.argv) > 1:
    PORT = int(sys.argv[1])
else:
    PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Increase kernel receive buffer to reduce drops under high packet rate.
try:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
except OSError:
    pass
sock.bind(("0.0.0.0", PORT))
print(f"[UDP] Listening on 0.0.0.0:{PORT}")

count = 0
buffers = {}
while True:
    data, addr = sock.recvfrom(65535)
    buf = buffers.setdefault(addr, bytearray())
    buf.extend(data)

    while True:
        if len(buf) < 28:
            break

        magic, width, height, payload_size, det_count, fmt, fid = struct.unpack('<7I', buf[:28])
        if magic != MAGIC:
            buf.clear()
            print(f"wrong magic from {addr}: {magic:#x}")
            break

        det_bytes = det_count * DET_SIZE
        prefix_len = 28 + det_bytes
        total_len = prefix_len + payload_size

        if len(buf) < total_len:
            break

        offset = 28
        dets = []
        for i in range(det_count):
            vals = struct.unpack(DET_STRUCT, buf[offset:offset + DET_SIZE])
            dets.append({
                'x': vals[0], 'y': vals[1], 'w': vals[2], 'h': vals[3], 'score': vals[4], 'target': vals[5]
            })
            offset += DET_SIZE

        jpeg = bytes(buf[prefix_len:prefix_len + payload_size])
        count += 1

        # Decode JPEG and display in window (no saving)
        if HAVE_CV2:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                # draw detections (x,y,w,h are normalized center-based)
                ih, iw = img.shape[:2]
                for i, d in enumerate(dets):
                    try:
                        cx = int(d['x'] * width)
                        cy = int(d['y'] * height)
                        w_pix = max(1, int(d['w'] * width))
                        h_pix = max(1, int(d['h'] * height))
                    except Exception:
                        # fallback to image size if header values are inconsistent
                        cx = int(d['x'] * iw)
                        cy = int(d['y'] * ih)
                        w_pix = max(1, int(d['w'] * iw))
                        h_pix = max(1, int(d['h'] * ih))

                    x0 = max(0, cx - w_pix // 2)
                    y0 = max(0, cy - h_pix // 2)
                    x1 = min(iw - 1, cx + w_pix // 2)
                    y1 = min(ih - 1, cy + h_pix // 2)

                    color = (0, 255, 0)
                    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                    label = f"target={int(d['target'])}"
                    tl = max(1, int(0.5 + 0.5 * (iw / 320)))
                    txt_y = y0 - 6 if y0 - 6 > 6 else y0 + 12
                    cv2.putText(img, label, (x0, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, tl)

                # overlay count
                cv2.putText(img, f"dets={len(dets)}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow('udp_receiver', img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Quit requested')
                    sys.exit(0)
            else:
                print(f"{count}: from {addr} size={payload_size} dets={len(dets)} fid={fid} -> decode failed")
        else:
            print(f"{count}: from {addr} size={payload_size} dets={len(dets)} fid={fid} (cv2 not available)")

        del buf[:total_len]
