# modules/vision.py
import cv2
import mss
import numpy as np
import ctypes
from ctypes import wintypes
import pygetwindow as gw
import config
import time
import pytesseract # pip install pytesseract

# === Tesseract 설치 경로 설정 (사용자 환경에 맞게 수정 필요) ===
# 윈도우 환경변수에 등록되어 있다면 주석 처리해도 됩니다.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === [좌표 계산용 구조체] ===
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
        
        # OCR 관련 변수
        self.last_ocr_time = 0
        self.current_kill_count = 0

    def find_maple_window(self):
        try:
            windows = gw.getWindowsWithTitle('MapleStory')
            if not windows:
                print("❌ 'MapleStory' 창을 찾을 수 없습니다.")
                return False
            win = windows[0]
            if win.isMinimized: win.restore()
            
            # 클라이언트 영역 정밀 계산
            rect = get_client_area_on_screen(win._hWnd)
            if not rect: return False
            
            x, y, w, h = rect
            self.capture_area = {"top": y, "left": x, "width": w, "height": h}
            self.window_found = True
            print(f"✅ 메이플 창 발견 (OCR 준비 완료): {self.capture_area}")
            return True
        except Exception as e:
            print(f"⚠️ 창 찾기 오류: {e}")
            return False
        
    def set_roi(self, rect):
        """GUI에서 지정한 ROI 영역 저장 (x, y, w, h)"""
        self.kill_roi = rect
        print(f"ROI 설정됨: {self.kill_roi}")

    def get_kill_count_ocr(self, frame):
        """ 지정된 영역만 잘라내서 숫자로 변환 """
        if not self.kill_roi:
            return self.current_kill_count
            
        # 0.5초에 한 번만 인식 (부하 감소)
        if time.time() - self.last_ocr_time < 0.5:
            return self.current_kill_count
        
        self.last_ocr_time = time.time()
        
        try:
            # 1. ROI 영역 자르기
            x, y, w, h = self.kill_roi
            # 프레임 범위를 벗어나지 않게 클리핑
            h_img, w_img = frame.shape[:2]
            if x < 0 or y < 0 or x+w > w_img or y+h > h_img:
                return self.current_kill_count
                
            roi = frame[y:y+h, x:x+w]
            
            # 2. 이미지 전처리 (인식률 높이기 핵심)
            # 흑백 변환
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 크기 확대 (작은 글씨 인식용, 3배 확대)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            # 이진화 (글자는 검정, 배경은 흰색, 혹은 그 반대로 확실하게 분리)
            # 메이플 전투분석창은 어두운 배경에 흰 글씨 -> 반전시켜서 흰 배경에 검은 글씨로 만듦 (Tesseract 선호)
            gray = cv2.bitwise_not(gray)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # 3. Tesseract 실행 (숫자 모드)
            # --psm 7: 이미지를 한 줄의 텍스트로 취급
            # digits: 숫자만 인식하도록 제한
            text = pytesseract.image_to_string(thresh, config='--psm 7 outputbase digits')
            
            # 4. 숫자 추출
            digits = ''.join(filter(str.isdigit, text))
            
            if digits:
                self.current_kill_count = int(digits)
                # print(f"인식된 숫자: {self.current_kill_count}") # 디버깅용
                
        except Exception as e:
            # print(f"OCR 에러: {e}")
            pass
            
        return self.current_kill_count

    def capture_and_analyze(self):
        if not self.window_found:
            if not self.find_maple_window():
                return np.zeros((100, 100, 3), dtype=np.uint8), 0, 0 # kill_count 추가
        
        try:
            with mss.mss() as sct:
                img_np = np.array(sct.grab(self.capture_area))
            
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            
            # 엔트로피 계산
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            entropy_score = np.sum(edges) / 255
            
            # OCR 수행
            kill_count = self.get_kill_count_ocr(frame)
            
            return frame, entropy_score, kill_count # 3개 반환
            
        except Exception as e:
            return np.zeros((100, 100, 3), dtype=np.uint8), 0, 0