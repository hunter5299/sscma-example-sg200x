# face_rtsp

RTSP streaming with on-device overlay (YOLO + Age/Gender/Race + Emotion). Boxes are drawn on the RTSP channel and sent out together with the stream.

## Run

```bash
./face_rtsp <yolo_face.cvimodel> <age_gender_race.cvimodel> <emotion.cvimodel> [single|multi] [threshold] [skip] [rtsp_port] [rtsp_session]

cd /home/recamera/sscma-example-sg200x/solutions/sesg-project/face_rtsp
sudo ./build/face_rtsp ./model/yolo-face_mixfp16.cvimodel ./model/age_gender_race_bf16.cvimodel ./model/emotion_bf16.cvimodel single 0.50 1 8554 live
```

Defaults: `rtsp_port=8554`, `rtsp_session=live`.

## Play

```text
rtsp://<device-ip>:<rtsp_port>/<rtsp_session>
```

Example:

```bash
ffplay -rtsp_transport tcp rtsp://<device-ip>:8554/live
```