#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONVIF 客户端 + RTSP 验收脚本 (PC 端运行)

完整演示一台 ONVIF AI IPC 的标准接入流程：
  1) WS-Discovery：UDP 组播 239.255.255.250:3702 发送 Probe，发现局域网内的 ONVIF 设备；
  2) SOAP：调用 GetDeviceInformation / GetProfiles / GetStreamUri 拿到设备信息和 RTSP 地址；
  3) RTSP：用 OpenCV(FFMPEG) 打开返回的 rtsp:// 地址，显示带 YOLO 检测框的实时画面。

仅依赖标准库 + opencv-python（RTSP 显示用）。不依赖 onvif-zeep 等第三方 ONVIF 库，
以便清楚展示协议本身。

示例：
  python3 onvif_client.py --discover                       # 仅发现设备
  python3 onvif_client.py --host 192.168.42.1 --onvif-port 8080   # 直连 SOAP 并拉流
  python3 onvif_client.py --discover --play                # 发现后自动拉流显示
"""

import argparse
import re
import socket
import struct
import sys
import time
import urllib.request

WSD_ADDR = "239.255.255.250"
WSD_PORT = 3702

PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header>
  <w:MessageID>uuid:{msgid}</w:MessageID>
  <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
  <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
 </e:Header>
 <e:Body>
  <d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>
 </e:Body>
</e:Envelope>"""


def discover(timeout=3.0):
    """发送 WS-Discovery Probe，收集 ProbeMatch 中的 XAddrs。"""
    msg = PROBE.format(msgid="a1b2c3d4-0000-0000-0000-000000000001").encode("utf-8")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    s.settimeout(timeout)

    print(f"[discover] 发送 Probe 到 {WSD_ADDR}:{WSD_PORT} ...")
    s.sendto(msg, (WSD_ADDR, WSD_PORT))

    found = {}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            data, addr = s.recvfrom(65535)
        except socket.timeout:
            break
        text = data.decode("utf-8", "ignore")
        xaddrs = re.findall(r"<[^>]*XAddrs>([^<]+)<", text)
        types = re.findall(r"<[^>]*Types>([^<]+)<", text)
        if xaddrs:
            url = xaddrs[0].strip().split()[0]
            found[url] = {"from": addr[0], "types": types[0] if types else ""}
            print(f"[discover] 发现设备 {addr[0]} -> {url}")
    s.close()
    return found


def soap_call(url, action_body, timeout=5.0):
    """对 ONVIF 服务发起一次 SOAP POST，返回响应文本。"""
    envelope = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
        'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
        'xmlns:tt="http://www.onvif.org/ver10/schema">'
        "<s:Body>" + action_body + "</s:Body></s:Envelope>"
    )
    req = urllib.request.Request(
        url,
        data=envelope.encode("utf-8"),
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def get_device_info(dev_url):
    body = "<tds:GetDeviceInformation/>"
    text = soap_call(dev_url, body)
    fields = {}
    for tag in ("Manufacturer", "Model", "FirmwareVersion", "SerialNumber", "HardwareId"):
        m = re.search(rf"<[^>]*{tag}>([^<]*)<", text)
        if m:
            fields[tag] = m.group(1)
    return fields


def get_stream_uri(media_url):
    body = (
        '<trt:GetStreamUri><trt:StreamSetup>'
        '<tt:Stream>RTP-Unicast</tt:Stream>'
        '<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>'
        '</trt:StreamSetup><trt:ProfileToken>profile0</trt:ProfileToken>'
        '</trt:GetStreamUri>'
    )
    text = soap_call(media_url, body)
    m = re.search(r"<[^>]*Uri>([^<]+)<", text)
    return m.group(1).strip() if m else None


def media_url_from_device(dev_url):
    """device_service URL -> media_service URL（本 demo 同端口同主机）。"""
    return dev_url.replace("device_service", "media_service")


def play_rtsp(url, transport="tcp", record_path=None, max_seconds=0):
    import os
    import cv2

    opts = f"rtsp_transport;{transport}|stimeout;5000000|max_delay;0"
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", opts)
    print(f"[rtsp] 打开 {url} (transport={transport})")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
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
                print("[rtsp] 打开失败，重试...")
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.time() - last_ok > 3.0:
                    print("[rtsp] 读帧超时，退出")
                    break
                time.sleep(0.01)
                continue
            last_ok = time.time()

            if record_path and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(record_path, fourcc, 25, (w, h))
                print(f"[rtsp] 录制到 {record_path} ({w}x{h})")
            if writer is not None:
                writer.write(frame)

            cv2.imshow("ONVIF + YOLO (RTSP)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if max_seconds and time.time() - t0 >= max_seconds:
                print(f"[rtsp] 达到 {max_seconds}s，停止")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="ONVIF client + RTSP verifier")
    ap.add_argument("--discover", action="store_true", help="执行 WS-Discovery 发现设备")
    ap.add_argument("--host", default="", help="直连设备 IP（跳过发现）")
    ap.add_argument("--onvif-port", type=int, default=8080, help="ONVIF SOAP 端口")
    ap.add_argument("--play", action="store_true", help="拿到 RTSP 地址后显示画面")
    ap.add_argument("--record", default="", help="录制 RTSP 到 mp4 文件")
    ap.add_argument("--seconds", type=int, default=0, help="录制/播放秒数（0=直到 q）")
    ap.add_argument("--transport", choices=["tcp", "udp"], default="tcp")
    ap.add_argument("--timeout", type=float, default=3.0, help="发现超时秒数")
    args = ap.parse_args()

    dev_url = None
    if args.discover:
        found = discover(args.timeout)
        if not found:
            print("[discover] 未发现任何 ONVIF 设备")
            if not args.host:
                return 1
        else:
            dev_url = next(iter(found))
    if args.host and not dev_url:
        dev_url = f"http://{args.host}:{args.onvif_port}/onvif/device_service"

    if not dev_url:
        print("请用 --discover 或 --host 指定设备")
        return 1

    print(f"\n[onvif] device_service = {dev_url}")
    try:
        info = get_device_info(dev_url)
        print("[onvif] 设备信息:")
        for k, v in info.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"[onvif] GetDeviceInformation 失败: {e}")

    media_url = media_url_from_device(dev_url)
    try:
        uri = get_stream_uri(media_url)
        print(f"[onvif] RTSP Stream URI = {uri}")
    except Exception as e:
        print(f"[onvif] GetStreamUri 失败: {e}")
        uri = None

    if uri and (args.play or args.record):
        play_rtsp(uri, args.transport, args.record or None, args.seconds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
