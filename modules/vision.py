# modules/vision.py
import cv2
import mss
import numpy as np
import ctypes
from ctypes import wintypes
import pygetwindow as gw
import config
import time
import pytesseract

# Tesseract 경로
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

# 좌표 계산용 구조체
user32 = ctypes.windll.user32
class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

def get_client_area_on_screen(hwnd):
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)): return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)): return None
    return pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top

class VisionSystem:
    def __init__(self):
        self.capture_area = config.DEFAULT_CAPTURE_AREA
        self.window_found = False
        
        self.kill_roi = None 
        self.minimap_roi = None # [신규] 미니맵 영역
        
        self.last_ocr_time = 0
        self.current_kill_count = 0
        self.player_pos = (0, 0)

    def find_maple_window(self):
        try:
            windows = gw.getWindowsWithTitle('MapleStory')
            if not windows:
                print("Error: 'MapleStory' window not found.")
                return False
            win = windows[0]
            if win.isMinimized: win.restore()
            
            rect = get_client_area_on_screen(win._hWnd)
            if not rect: return False
            
            x, y, w, h = rect
            self.capture_area = {"top": y, "left": x, "width": w, "height": h}
            self.window_found = True
            print(f"Maple Window Found: {self.capture_area}")
            return True
        except Exception as e:
            print(f"Window Find Error: {e}")
            return False

    def set_roi(self, rect):
        self.kill_roi = rect
        print(f"Kill ROI Set: {self.kill_roi}")

    def set_minimap_roi(self, rect):
        """[신규] 미니맵 영역 설정"""
        self.minimap_roi = rect
        print(f"Minimap ROI Set: {self.minimap_roi}")

    def get_player_position(self, frame):
        """[신규] 미니맵 ROI 안에서 노란색(내 캐릭터) 위치 찾기"""
        if not self.minimap_roi:
            return 0, 0
            
        x, y, w, h = self.minimap_roi
        h_img, w_img = frame.shape[:2]
        if x < 0 or y < 0 or x+w > w_img or y+h > h_img:
            return 0, 0
            
        # 1. 미니맵만 잘라내기
        minimap_img = frame[y:y+h, x:x+w]
        
        # 2. 노란색 추출 (HSV 색상 공간 사용)
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
        
        # 메이플 미니맵의 노란색 범위
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 3. 무게중심(Centroid) 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.player_pos = (cx, cy)
                return cx, cy
        
        return self.player_pos

    def get_kill_count_ocr(self, frame):
        if not self.kill_roi:
            return self.current_kill_count
            
        if time.time() - self.last_ocr_time < 0.5:
            return self.current_kill_count
        
        self.last_ocr_time = time.time()
        
        try:
            x, y, w, h = self.kill_roi
            h_img, w_img = frame.shape[:2]
            if x < 0 or y < 0 or x+w > w_img or y+h > h_img:
                return self.current_kill_count
                
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_not(thresh)
            
            config_tess = '--psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(thresh, config=config_tess)
            text_clean = text.strip()
            
            if text_clean.isdigit():
                self.current_kill_count = int(text_clean)
                # print(f"Kill Count: {self.current_kill_count}")
                
        except Exception as e:
            pass
            
        return self.current_kill_count

    def capture_and_analyze(self):
        if not self.window_found:
            if not self.find_maple_window():
                return np.zeros((100, 100, 3), dtype=np.uint8), 0, 0, 0, 0
        
        try:
            with mss.mss() as sct:
                img_np = np.array(sct.grab(self.capture_area))
            
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            entropy_score = np.sum(edges) / 255
            
            kill_count = self.get_kill_count_ocr(frame)
            px, py = self.get_player_position(frame) # [신규] 좌표 반환
            
            return frame, entropy_score, kill_count, px, py 
            
        except Exception as e:
            print(f"Capture Error: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8), 0, 0