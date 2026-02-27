jpge_delay
==========

简介
----
`jpge_delay` 用于做一次 JPEG 往返时延测试：

1) 设备端发送 1 张 JPEG（udp_service 格式）
2) 立即开始计时并等待回传
3) 收到完整 JPEG 后停止计时、打印耗时并保存图片，程序结束

配套脚本 `echo_jpeg_udp.py` 在主机端接收 JPEG、展示，然后原封不动按相同格式回传。

设备端用法
---------
默认目标地址与端口：`192.168.2.101:5001`，本地回传接收端口：`5002`。

```bash
# 设备端
sudo ./jpge_delay

# 自定义：发送到 192.168.2.200:5001，并监听 5002 等待回传
sudo ./jpge_delay 192.168.2.200 5001 5002
```

主机端用法（Python 回传）
----------------------
```bash
# 主机端：监听 5001，回传到设备 IP:5002
python3 echo_jpeg_udp.py --bind 0.0.0.0 --port 5001 --reply-ip <设备IP> --reply-port 5002
```

说明
----
- 回传数据使用与发送端一致的 udp_service 头部与 JPEG payload。
- 设备端保存文件：`echo_frame_<fid>.jpg`。
- `echo_jpeg_udp.py` 需要 OpenCV 才能显示（未安装时会自动跳过显示）。
