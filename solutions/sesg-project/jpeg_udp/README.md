jpeg_udp
========

简介
----
`jpeg_udp` 是一个轻量的可执行程序，用于采集摄像头上的 JPEG 帧并通过 UDP 发送（使用工程内的 `SesgJpegUdpStreamer` / `udp_service`）。

默认目标地址：`192.168.2.101:5001`。默认全速发送。运行时可通过位置参数设置“发送间隔秒数”（`interval_s`）：

- `./jpeg_udp`                         — 默认地址、全速发送
- `./jpeg_udp <interval_s>`            — 默认地址、按间隔发送（如 `1`、`2`、`0.1`）
- `./jpeg_udp <ip> <port>`             — 指定 `ip:port`、全速发送
- `./jpeg_udp <ip> <port> <interval_s>`— 指定 `ip:port`、按间隔发送

协议说明
--------
`udp_service` 采用统一的帧格式（所有设备端发送端/接收端遵循）：

头部（全部为小端，32-bit 单位）：
- magic (u32)
- width (u32)
- height (u32)
- payload_size (u32)   // JPEG 或 RGB payload 的字节数
- det_count (u32)
- fmt (u32)            // payload 格式：`1` 表示 JPEG
- fid (u32)            // 帧ID

紧跟可选的 `det_bytes`（长度 = det_count * sizeof(T)）
然后是 payload bytes（长度 = payload_size）。

接收端需要合并/重组来自多包的 payload（`udp_service` 会负责把大 payload 分片到多 UDP 包），并解析上述头部再保存/解码 JPEG。

Python 接收示例
-------------
仓库中包含 `recv_jpeg_udp.py`：一个用于在主机上监听并保存连续收到的 JPEG 帧的示例脚本。脚本会：

- 在指定 UDP 端口监听
- 按协议解析统一头
- 重组分片（脚本内做了基本的按 `fid` 的重组）
- 将完整的 JPEG payload 写入文件 `frame_<fid>.jpg`

使用示例：

```bash
# 在接收端运行（主机）
python3 recv_jpeg_udp.py --bind 0.0.0.0 --port 5001 --outdir /tmp/recv_jpegs

# 在设备端运行 jpeg_udp（按默认地址，全速）
sudo ./jpeg_udp

# 或指定目标
sudo ./jpeg_udp 192.168.2.200 5001

# 或按间隔发送（1秒一张）
sudo ./jpeg_udp 1

# 指定目标并按间隔发送（0.1秒一张）
sudo ./jpeg_udp 192.168.2.200 5001 0.1
```

