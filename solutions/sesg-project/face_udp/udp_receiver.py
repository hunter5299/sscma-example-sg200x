#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能 UDP 接收端 (PC端运行)
功能：接收来自 SG2002 C++ 程序的 UDP 数据包 (JPEG + 检测结果) 并显示
依赖：仅需 pip install opencv-python numpy
"""

import socket
import struct
import numpy as np
import cv2
import threading
import queue
import time

# ==================== 配置 ====================
LISTEN_IP = "0.0.0.0"      # 监听所有网卡
LISTEN_PORT = 5001         # 必须与 C++ 端的端口一致
BUFFER_SIZE = 10 * 1024 * 1024  # UDP 接收缓冲
MAGIC_NUMBER = 0xFACEBEEF
DISPLAY_SCALE = 1.0        # 显示缩放倍数 (1.0 = 原图大小)
DISPLAY_SIZE = (640, 640)  # 强制显示分辨率 (宽,高)

# ================ 左上角汇总文字配置 (方便调整) ================
# 如果识别到多个人，会按行依次绘制；行间距由 SUMMARY_LINE_HEIGHT 控制。
# 你只需要修改下面这几个常量即可：
SUMMARY_LINE_HEIGHT = 22        # 行间距，像素；两行文字的垂直间隔就在这里调整
SUMMARY_FONT_SCALE = 0.65     # 字体缩放因子，传给 cv2.putText
SUMMARY_THICKNESS = 2         # 字体笔画粗细（像素）
SUMMARY_COLOR = (0, 0, 255)   # BGR 颜色元组，红色

# 备注：如果你觉得文字仍然不够清晰，可把 SUMMARY_THICKNESS 调大到 3，
# 或把 SUMMARY_FONT_SCALE 适当增大，并保持 resize 到 DISPLAY_SIZE 再绘制。

# ================ 解析属性说明 (模型/协议字段) ================
# 下面列出 UDP 包里每个 Detection 中包含的四个属性及其取值含义和中文翻译：
# 1) gender (性别)
#    - 数值含义：1 = Male (男), 0 = Female (女), -1 = Unknown (未知)
#    - 用途：用于显示与统计
# 2) age (年龄档位)
#    - 模型输出为 9 档 age bin (0..8)，对应范围：
#      0: 0-2 (0-2岁)
#      1: 3-9 (3-9岁)
#      2: 10-19 (10-19岁)
#      3: 20-29 (20-29岁)
#      4: 30-39 (30-39岁)
#      5: 40-49 (40-49岁)
#      6: 50-59 (50-59岁)
#      7: 60-69 (60-69岁)
#      8: 70+ (70岁及以上)
# 3) race (种族/人群)
#    - 模型/协议常用顺序及中文说明：
#      0: White (白人)
#      1: Black (黑人)
#      2: Latino_Hispanic (拉丁/西班牙裔)
#      3: East_Asian (东亚人)
#      4: Southeast_Asian (东南亚人)
#      5: Indian (印度人)
#      6: Middle_Eastern (中东人)
# 4) emotion (表情)
#    - Emotion id (0..6) 对应：
#      0: angry (生气)
#      1: disgust (厌恶)
#      2: fear (害怕)
#      3: happy (高兴)
#      4: sad (难过)
#      5: surprise (惊讶)
#      6: neutral (中性)
# 其它说明：
# - *_score 字段表示对应属性的置信度（通常 0..1），可用于过滤或排序。
# - target 字段是人脸索引（例如 face0, face1），用于与设备端日志对齐。
# - 如果模型/协议将来更改，请同时更新本注释与下面的映射函数（_gender_label/_age_bin_label/_race_label/emo_labels）。

# ==================== 数据结构 ====================
class FrameRawData:
    """仅存储原始字节流，不解码"""
    def __init__(self, width, height, detections, img_bytes, fid):
        self.width = width
        self.height = height
        self.detections = detections
        self.img_bytes = img_bytes
        self.fid = fid

    # 说明：
    # - width,height: 原始 JPEG 图（未 resize）的宽高（整数）。
    # - detections: Detection 对象列表，按 target 索引排序。
    # - img_bytes: JPEG 二进制数据（bytes）。接收端使用 numpy.frombuffer + cv2.imdecode 解码。
    # - fid: 设备端帧 ID，便于排查丢帧/乱序问题。

class Detection:
    __slots__ = [
        'x', 'y', 'w', 'h', 'score',
        'target', 'gender', 'age', 'race', 'emotion',
        'gender_score', 'age_score', 'race_score', 'emotion_score'
    ]
    def __init__(self, data_tuple):
        (
            self.x, self.y, self.w, self.h, self.score,
            self.target, self.gender, self.age, self.race, self.emotion,
            self.gender_score, self.age_score, self.race_score, self.emotion_score
        ) = data_tuple

    # 说明：
    # - x,y,w,h,score 为相对坐标/大小（0..1），必须乘以渲染图的宽高得到像素坐标。
    # - target: 人脸序号（用于和设备端输出对齐，例如 face0, face1）。
    # - gender/age/race/emotion: 属性 id，数值意义由模型/协议决定；接收端用 label 函数映射成字符串显示。
    # - *_score: 对应属性的置信度（0..1）。


def _gender_label(g: int) -> str:
    # 与 C++ 发送端一致：1=Male, 0=Female, -1=Unknown
    if g == 1:
        return "Male"
    if g == 0:
        return "Female"
    return "N/A"


# 说明：下面的 label helper 都是把模型/协议里的数值 id 映射成人类可读的字符串。
# 如果模型或协议更改，这里需要同步修改映射表。


def _age_bin_label(a: int) -> str:
    # FairFace 常见 9 档：0..8
    # 0:0-2, 1:3-9, 2:10-19, 3:20-29, 4:30-39, 5:40-49, 6:50-59, 7:60-69, 8:70+
    bins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    if 0 <= a < len(bins):
        return bins[a]
    return "N/A"


def _race_label(r: int) -> str:
    # FairFace label 顺序
    labels = [
        "White",
        "Black",
        "Latino_Hispanic",
        "East_Asian",
        "Southeast_Asian",
        "Indian",
        "Middle_Eastern",
    ]
    if 0 <= r < len(labels):
        return labels[r]
    return "N/A"


def _is_jpeg_soi(buf: bytearray, offset: int) -> bool:
    if offset + 2 > len(buf):
        return False
    return buf[offset] == 0xFF and buf[offset + 1] == 0xD8

# ==================== UDP 接收器 类说明 ====================
# UDPReceiver 负责：
# - 在独立线程中使用 recv_into 往内存缓冲区追加 UDP 数据（减少内存分配）
# - 在 parse_packets() 中尽可能做无拷贝解析：通过 struct.unpack_from 读取 header/det，
#   并用 buffer 切片得到 JPEG bytes（随后转换为 bytes 存入最新帧，保证线程安全）
# - 支持两种 FaceResult 长度（40/56 bytes），并通过检查 JPEG SOI 来自动判定，避免解包错位
# 注意事项：
# - buffer 是一个固定长度的 bytearray；当剩余空间不足时会调用 compact_buffer 将未读数据移到头部。
# - 解析发现魔数错位时，会逐字节滑动查找魔数，兼容网络流中的边界情况。

# ==================== UDP 接收器 ====================
class UDPReceiver:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.running = False
        
        # 线程锁保护的最新帧容器 (只存最新一帧，解决延迟问题)
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # 接收缓冲区
        self.buffer = bytearray(BUFFER_SIZE)
        self.write_pos = 0 
        self.parse_pos = 0 
        
    def init_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 增大操作系统内核缓冲区，防止丢包
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)
        print(f"[UDP] 正在监听 {self.ip}:{self.port} ...")

    def compact_buffer(self):
        """内存搬运：把未处理的数据移到头部"""
        # 解释：当 parse_pos 前移导致缓冲区碎片（前面是已读数据）时，
        # 把剩余未读数据移动到缓冲区开头，重置指针以便继续读入。
        if self.parse_pos > 0:
            remain = self.write_pos - self.parse_pos
            if remain > 0:
                # 使用切片搬运数据（Python 内部会做 memmove）
                self.buffer[0:remain] = self.buffer[self.parse_pos:self.write_pos]
            self.write_pos = remain
            self.parse_pos = 0

    def parse_packets(self):
        """解析缓冲区中的完整帧"""
        # 协议头大小: Magic(4) + Header(20) + Fid(4) = 28 bytes
        HEADER_SIZE = 28
        # 备注：FaceResult 在仓库里存在两种版本：
        # - 56 bytes: <fffffiiiiiffff (x,y,w,h,score,target,gender,age,race,emotion,4xscore)
        # - 40 bytes: <fffffiiiff      (x,y,w,h,score,target,gender,age,2xscore)
        # 为了避免用户跑错脚本导致“结果永远不对”，这里自动判定。
        DET56_SIZE = 56
        DET40_SIZE = 40
        
        # 循环解析尽可能多的完整帧；当数据不够时跳出等待更多 UDP 包。
        while self.write_pos - self.parse_pos >= HEADER_SIZE:
            # 1. 寻找魔数
            # 读取 4 字节魔数（小端）
            magic = struct.unpack_from('<I', self.buffer, self.parse_pos)[0]
            if magic != MAGIC_NUMBER:
                # 若魔数不匹配，说明缓冲区当前位置不是帧起始，逐字节滑动以寻找下一个可能的帧。
                self.parse_pos += 1
                continue

            # 2. 解析头部
            # Header: w, h, img_size, det_count, fmt
            header = struct.unpack_from('<IIIII', self.buffer, self.parse_pos + 4)
            width, height, img_size, det_count, fmt = header
            
            # Fid
            fid = struct.unpack_from('<I', self.buffer, self.parse_pos + 24)[0]

            # 备注：选择 det_size：优先用 JPEG 起始标记 (0xFFD8) 验证偏移。
            # 只有当缓冲区里数据足够时才检查 SOI，否则可能越界。
            det_size = None
            total_len_56 = HEADER_SIZE + (det_count * DET56_SIZE) + img_size
            total_len_40 = HEADER_SIZE + (det_count * DET40_SIZE) + img_size

            have_56 = (self.write_pos - self.parse_pos) >= total_len_56
            have_40 = (self.write_pos - self.parse_pos) >= total_len_40

            if have_56 and _is_jpeg_soi(self.buffer, self.parse_pos + HEADER_SIZE + det_count * DET56_SIZE):
                det_size = DET56_SIZE
            elif have_40 and _is_jpeg_soi(self.buffer, self.parse_pos + HEADER_SIZE + det_count * DET40_SIZE):
                det_size = DET40_SIZE
            elif have_56 and not have_40:
                det_size = DET56_SIZE
            elif have_40 and not have_56:
                det_size = DET40_SIZE
            elif have_56 and have_40:
                # 两者都“长度足够但 SOI 不匹配”，兜底优先 56（face_udp 正式协议）
                det_size = DET56_SIZE
            else:
                # 数据不够，等待下次接收
                break

            total_len = HEADER_SIZE + (det_count * det_size) + img_size
            
            # 3. 数据不够，等待下次接收
            if self.write_pos - self.parse_pos < total_len:
                break 

            # 4. 提取数据
            current_offset = self.parse_pos + HEADER_SIZE
            detections = []
            
            # 解析检测框
            # 解析每个 Detection 记录到 Detection 对象
            if det_size == DET56_SIZE:
                fmt_det = '<fffffiiiiiffff'
                for _ in range(det_count):
                    det_data = struct.unpack_from(fmt_det, self.buffer, current_offset)
                    detections.append(Detection(det_data))
                    current_offset += DET56_SIZE
            else:
                # 40 bytes 版本缺少 race/emotion，这里补默认值
                fmt_det = '<fffffiiiff'
                for _ in range(det_count):
                    x, y, w, h, score, target, gender, age, gender_score, age_score = struct.unpack_from(
                        fmt_det, self.buffer, current_offset
                    )
                    det_data = (
                        x, y, w, h, score,
                        target, gender, age, -1, -1,
                        gender_score, age_score, 0.0, 0.0
                    )
                    detections.append(Detection(det_data))
                    current_offset += DET40_SIZE
            
            # 提取图片 (Zero Copy slice)
            # img_bytes 直接使用 buffer 的切片（Zero Copy 语义），随后转换为 bytes 存储到 FrameRawData
            img_bytes = self.buffer[current_offset : current_offset + img_size]
            
            # 5. 更新最新帧 (覆盖旧的)
            # 将解析到的帧写入 self.latest_frame，使用 bytes() 做一次拷贝以断开对底层 buffer 的引用，
            # 否则 buffer 后续被复用会导致已保存的 jpeg 数据被篡改。
            with self.lock:
                self.latest_frame = FrameRawData(width, height, detections, bytes(img_bytes), fid)

            # 6. 移动指针
            self.parse_pos += total_len

    def recv_loop(self):
        print("[接收线程] 启动")
        MAX_PACKET = 65535
        while self.running:
            try:
                # 缓冲区快满了就整理一下
                if len(self.buffer) - self.write_pos < MAX_PACKET:
                    self.compact_buffer()
                
                # 接收数据
                nbytes = self.sock.recv_into(memoryview(self.buffer)[self.write_pos:])
                if nbytes > 0:
                    self.write_pos += nbytes
                    self.parse_packets()
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[UDP Error] {e}")
                self.parse_pos = 0
                self.write_pos = 0

    # 注意：recv_loop 使用 recv_into(memoryview(self.buffer)[self.write_pos:]) 以避免每次接收都分配新的 bytes 对象，
    # 这在高帧率下能显著减少内存压力。但同时要求 parse_packets 必须及时把已解析的数据从 buffer 中移动或丢弃，
    # 否则会导致 buffer 被耗尽。

    def start(self):
        self.running = True
        self.init_socket()
        t = threading.Thread(target=self.recv_loop, daemon=True)
        t.start()
        return t

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
    
    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

# ==================== 渲染器 ====================
class Renderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def process(self, raw_frame):
        if raw_frame is None: return None

        # 1. 解码 JPEG
        try:
            img_array = np.frombuffer(raw_frame.img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: return None
        except:
            return None

        # 备注（清晰度与绘制顺序）：
        # - 先 resize 到 DISPLAY_SIZE 再绘制文字，可以避免绘制后缩放导致的文字模糊（插值问题）。
        # - 如果你希望在原始分辨率上绘制并再缩放，可能会看到文字被模糊。
        # - 这里默认按 DISPLAY_SIZE 缩放后再绘制，以保证文字清晰度。
        if DISPLAY_SIZE is not None:
            img = cv2.resize(img, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

        # 2. 画框
        h, w = img.shape[:2]

        emo_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # 文字统一汇总到左上角
        summary_lines = []

        for det in raw_frame.detections:
            # 反归一化坐标
            x_center, y_center = det.x * w, det.y * h
            box_w, box_h = det.w * w, det.h * h
            
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # gender: 1=Male, 0=Female, -1=Unknown
            gender_str = _gender_label(det.gender)
            age_str = _age_bin_label(det.age)
            race_str = _race_label(det.race)
            emo_str = emo_labels[det.emotion] if 0 <= det.emotion < len(emo_labels) else "N/A"

            summary_lines.append(
                # 备注：按用户要求把 gender/age 放在最前面，便于一眼看懂。
                f"face{det.target}: {gender_str} age={age_str} race={race_str} emo={emo_str}"
            )
            color = (0, 255, 0) # 绿色框

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 备注：左上角绘制汇总文字（无背景）
        # - 使用顶部常量 SUMMARY_LINE_HEIGHT/SUMMARY_FONT_SCALE/SUMMARY_THICKNESS/SUMMARY_COLOR 控制样式。
        # - 两行文字的垂直间距就是 SUMMARY_LINE_HEIGHT，若要调整只需修改文件顶部的 SUMMARY_LINE_HEIGHT 常量。
        if summary_lines:
            x0, y0 = 10, 20
            for i, s in enumerate(summary_lines):
                y = y0 + i * SUMMARY_LINE_HEIGHT
                # 使用 LINE_AA 抗锯齿以获得更平滑的字符边缘
                cv2.putText(img, s, (x0, y), self.font, SUMMARY_FONT_SCALE, SUMMARY_COLOR, SUMMARY_THICKNESS, cv2.LINE_AA)
            
        return img

# ==================== 主程序 ====================
def main():
    receiver = UDPReceiver(LISTEN_IP, LISTEN_PORT)
    renderer = Renderer()
    receiver.start()

    print("等待数据中... 按 'q' 退出")
    
    try:
        while True:
            # 获取最新数据 (非阻塞)
            raw_frame = receiver.get_latest_frame()
            
            if raw_frame is None:
                # 没数据时短暂休眠，避免死循环占满 CPU
                time.sleep(0.01)
                continue

            # 输出已放到画面左上角；这里避免刷屏不再每帧 print
            
            # 渲染显示
            frame = renderer.process(raw_frame)
            if frame is not None:
                cv2.imshow("UDP Face Receiver", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()