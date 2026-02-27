#!/usr/bin/env python3
"""
Receive one UDP JPEG frame (udp_service format), display it, and echo back
the same JPEG payload using the same header format, then exit.
"""

import argparse
import socket
import struct
import time

HDR_FMT = '<7I'
HDR_SIZE = struct.calcsize(HDR_FMT)

class FrameReassembly:
    def __init__(self, total_size, fid):
        self.total_size = total_size
        self.fid = fid
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
        return b''.join(self.chunks)[: self.total_size]


def send_fragmented(sock, addr, header, payload, max_payload=1400):
    for off in range(0, len(payload), max_payload):
        chunk = payload[off: off + max_payload]
        pkt = struct.pack(HDR_FMT, *header) + chunk
        sock.sendto(pkt, addr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bind', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5001)
    p.add_argument('--reply-ip', default='192.168.2.11')
    p.add_argument('--reply-port', type=int, default=5002)
    p.add_argument('--show', action='store_true')
    p.add_argument('--no-show', action='store_true')
    args = p.parse_args()

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
    print(f'Listening on {args.bind}:{args.port}')

    reasm = {}

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue

            if len(data) < HDR_SIZE:
                continue

            hdr = struct.unpack(HDR_FMT, data[:HDR_SIZE])
            magic, width, height, payload_size, det_count, fmt, fid = hdr
            payload = data[HDR_SIZE:]

            if payload_size == 0 or fmt != 1:
                continue

            if len(payload) >= payload_size:
                jpeg = payload[:payload_size]
            else:
                if fid not in reasm:
                    reasm[fid] = FrameReassembly(payload_size, fid)
                reasm[fid].add_chunk(payload)
                if not reasm[fid].complete():
                    continue
                jpeg = reasm[fid].get_payload()
                del reasm[fid]

            if enable_show and cv2 is not None and np is not None:
                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow('jpge_delay', img)
                    cv2.waitKey(1)

            reply_header = (magic, width, height, payload_size, det_count, fmt, fid)
            send_fragmented(sock, (args.reply_ip, args.reply_port), reply_header, jpeg)
            print(f'Echoed fid={fid} size={len(jpeg)} -> {args.reply_ip}:{args.reply_port}')
            break

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == '__main__':
    main()
