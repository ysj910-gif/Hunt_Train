import time
import serial
import joblib
import pandas as pd
import cv2
import numpy as np
import threading
import mss
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import config

# 윈도우 좌표 계산용 (GUI와 동일한 로직 사용)
user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

def get_client_area_on_screen(hwnd):
    rect = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top

class ArduinoController:
    def __init__(self):
        self.ser = None
        self.pressed_keys = set()

    def connect(self, port, baudrate=115200):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2) 
            return True
        except Exception as e:
            print(f"Arduino Connection Error: {e}")
            return False

    def send_command(self, action, key):
        if self.ser is None: return
        try:
            self.ser.write(f"{action}{key}\n".encode())
        except: pass

    def update_keys(self, target_keys_str):
        if target_keys_str == 'None' or pd.isna(target_keys_str):
            new_keys = set()
        else:
            new_keys = set(target_keys_str.split('+'))

        for k in list(self.pressed_keys):
            if k not in new_keys:
                self.send_command('R', k)
                self.pressed_keys.remove(k)
        
        for k in new_keys:
            if k not in self.pressed_keys:
                self.send_command('P', k)
                self.pressed_keys.add(k)

    def release_all(self):
        if self.ser:
            try:
                self.ser.write(b'S\n')
                self.pressed_keys.clear()
            except: pass
    
    def close(self):
        if self.ser:
            self.release_all()
            self.ser.close()
            self.ser = None

class BotRunner:
    def __init__(self):
        self.arduino = ArduinoController()
        self.model = None
        self.is_running = False
        self.thread = None
        self.vision = None
        self.offset_x = 0
        self.offset_y = 0

    def load_model(self, path):
        try:
            self.model = joblib.load(path)
            return True, "모델 로드 성공"
        except Exception as e:
            return False, str(e)

    def connect_arduino(self, port):
        if self.arduino.connect(port):
            return True, "아두이노 연결 성공"
        return False, "연결 실패"

    def start(self, vision_engine, offset_x, offset_y):
        if self.is_running: return
        if not self.model or not self.arduino.ser:
            raise Exception("모델 또는 아두이노가 준비되지 않았습니다.")

        self.vision = vision_engine
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.arduino.release_all()

    def _loop(self):
        with mss.mss() as sct:
            while self.is_running:
                try:
                    # 1. 화면 캡처 (GUI와 동일한 로직)
                    windows = gw.getWindowsWithTitle('MapleStory')
                    if not windows:
                        time.sleep(1); continue
                    
                    win = windows[0]
                    c_left, c_top, c_w, c_h = get_client_area_on_screen(win._hWnd)
                    
                    capture_roi = {
                        "top": c_top + config.MINIMAP_ROI['top'],
                        "left": c_left + config.MINIMAP_ROI['left'],
                        "width": config.MINIMAP_ROI['width'],
                        "height": config.MINIMAP_ROI['height']
                    }
                    
                    img = np.array(sct.grab(capture_roi))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    # 2. 정보 추출
                    mask = self.vision.get_character_mask(frame)
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        raw_x = int(M["m10"] / M["m00"])
                        raw_y = int(M["m01"] / M["m00"])
                    else:
                        raw_x, raw_y = 0, 0
                    
                    # 보정된 좌표 사용
                    player_x = raw_x - self.offset_x
                    player_y = raw_y - self.offset_y

                    # 엔트로피 계산
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist_norm = hist.ravel() / hist.sum()
                    hist_norm = hist_norm[hist_norm > 0]
                    entropy = - (hist_norm * np.log2(hist_norm)).sum() * 10000

                    # 3. 예측 및 아두이노 전송
                    # 학습 때와 동일한 Feature 순서여야 함
                    features = pd.DataFrame([[player_x, player_y, entropy]], 
                                          columns=['player_x', 'player_y', 'entropy'])
                    action = self.model.predict(features)[0]
                    self.arduino.update_keys(action)

                    # 디버그용 (옵션)
                    # print(f"Pos: {player_x},{player_y} Act: {action}")

                    time.sleep(0.05) # 약 20 FPS

                except Exception as e:
                    print(f"Bot Loop Error: {e}")
                    time.sleep(1)
        
        self.arduino.release_all()