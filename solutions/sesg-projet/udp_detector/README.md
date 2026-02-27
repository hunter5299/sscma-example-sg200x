UDP Detector (YOLO only)

Overview

This project captures camera frames, runs YOLO detection, and streams JPEG frames plus detection results over UDP.

UDP Detector (YOLO only)

Run (sender)

```bash
sudo ./model_detector ./model/yolo11n_cv181x_int8.cvimodel 
# defaults: threshold=0.5 udp_ip=192.168.2.101 udp_port=5000
```

Run (receiver)

```
python3 ./tools/udp_receiver.py
```

