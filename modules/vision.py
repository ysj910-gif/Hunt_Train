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

    def get_kill_count_ocr(self, frame):
        """ '몬스터 처치' 텍스트 옆의 숫자를 인식 """
        # 성능을 위해 1초에 한 번만 실행
        if time.time() - self.last_ocr_time < 1.0:
            return self.current_kill_count
        
        self.last_ocr_time = time.time()
        
        try:
            # 1. 이미지 전처리 (흑백 변환 -> 이진화)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 글자가 잘 보이게 이진화 (흰 글씨/어두운 배경 가정)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # 2. 텍스트 데이터 추출 (한글 포함)
            # psm 11: 텍스트가 흩어져 있다고 가정 (Sparse text)
            data = pytesseract.image_to_data(thresh, lang='kor', config='--psm 11', output_type=pytesseract.Output.DICT)
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                
                # 3. 키워드 찾기 ("처치" 또는 "마리")
                if "처치" in text or "몬스터" in text:
                    # 키워드 발견! 그 뒤에 나오는 숫자 찾기 (최대 3단어 뒤까지 탐색)
                    for j in range(i + 1, min(i + 5, n_boxes)):
                        next_text = data['text'][j].strip()
                        # 숫자만 있거나 "123마리" 처럼 숫자가 포함된 경우
                        digits = ''.join(filter(str.isdigit, next_text))
                        if digits:
                            self.current_kill_count = int(digits)
                            # print(f"OCR 인식: {self.current_kill_count} 마리") # 디버깅용
                            return self.current_kill_count
                            
        except Exception as e:
            # print(f"OCR 오류 (Tesseract 설치 확인 필요): {e}")
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