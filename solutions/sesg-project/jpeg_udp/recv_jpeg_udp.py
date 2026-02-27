#!/usr/bin/env python3
"""
Simple UDP JPEG receiver for the project's `udp_service` format.

Usage:
  python3 recv_jpeg_udp.py --bind 0.0.0.0 --port 5001 --outdir ./recv

This is a minimal example: it performs basic reassembly by frame id (fid)
and writes `frame_<fid>.jpg` for each complete payload (fmt==1).

Note: The sender may split a large JPEG into multiple UDP packets; this
script expects each packet to contain a header with the same framing used
in `udp_service` and includes a simple reassembly cache keyed by `fid`.
"""

import argparse
import os
import socket
import struct
import time
from collections import defaultdict

# Header layout (all little-endian unsigned 32-bit):
# magic, width, height, payload_size, det_count, fmt, fid
HDR_FMT = '<7I'
HDR_SIZE = struct.calcsize(HDR_FMT)

# A simple reassembly structure per fid
class FrameReassembly:
    def __init__(self, total_size):
        self.total_size = total_size
        self.chunks = []
        self.received = 0
        self.last_ts = time.time()

    def add_chunk(self, data):
        self.chunks.append(data)
        self.received += len(data)
        self.last_ts = time.time()

    def complete(self):
        return self.received >= self.total_size

    def get_payload(self):
        return b''.join(self.chunks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bind', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5001)
    p.add_argument('--outdir', default='recv_jpegs')
    p.add_argument('--timeout', type=int, default=5, help='seconds to keep incomplete frames')
    p.add_argument('--show', action='store_true', help='display received frames using OpenCV')
    p.add_argument('--no-show', action='store_true', help='disable automatic display')
    p.add_argument('--save', action='store_true', help='save received frames to outdir')
    args = p.parse_args()
    if args.save:
        os.makedirs(args.outdir, exist_ok=True)

    # optional OpenCV display
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        cv2 = None
        np = None

    if args.no_show:
        enable_show = False
    elif args.show:
        enable_show = True
    else:
        enable_show = (cv2 is not None and np is not None)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(1.0)
    print('Listening on %s:%d' % (args.bind, args.port))

    reasm = {}  # fid -> FrameReassembly

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                # cleanup stale reassemblies
                now = time.time()
                stale = [fid for fid, r in reasm.items() if now - r.last_ts > args.timeout]
                for fid in stale:
                    print('Dropping stale fid', fid)
                    del reasm[fid]
                continue

            if len(data) < HDR_SIZE:
                print('Packet too small from', addr)
                continue

            hdr = struct.unpack(HDR_FMT, data[:HDR_SIZE])
            magic, width, height, payload_size, det_count, fmt, fid = hdr

            payload = data[HDR_SIZE:]

            # basic validation
            if payload_size == 0:
                continue

            # fmt==1 -> JPEG
            if fmt != 1:
                print('Unsupported fmt', fmt, 'from', addr)
                continue

            # If payload fits a single packet
            if len(payload) >= payload_size:
                jpeg = payload[:payload_size]
            else:
                # Partial: need reassembly
                if fid not in reasm:
                    reasm[fid] = FrameReassembly(payload_size)
                reasm[fid].add_chunk(payload)
                if not reasm[fid].complete():
                    continue
                jpeg = reasm[fid].get_payload()[:payload_size]
                del reasm[fid]

            # display if possible
            if enable_show and cv2 is not None and np is not None:
                try:
                    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imshow('jpeg_udp', img)
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            raise KeyboardInterrupt
                except Exception as e:
                    print('Display error:', e)

            # save only when requested
            if args.save:
                out_path = os.path.join(args.outdir, f'frame_{fid}.jpg')
                with open(out_path, 'wb') as f:
                    f.write(jpeg)
                print(f'Received fid={fid} size={len(jpeg)} -> {out_path}')
            else:
                print(f'Received fid={fid} size={len(jpeg)}')

    except KeyboardInterrupt:
        print('\nExiting')
    finally:
        sock.close()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == '__main__':
    main()
