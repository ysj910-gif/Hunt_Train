# modules/vision.py
import cv2
import mss
import numpy as np
import ctypes
from ctypes import wintypes
import pygetwindow as gw
import time
import pytesseract

# Tesseract 경로 (환경에 맞게 수정)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# DPI 인식 설정 (좌표 밀림 방지)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), 
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

def get_client_area_on_screen(hwnd):
    """
    진단 도구와 동일한 로직:
    창의 테두리(Title Bar 등)를 제외한 실제 게임 화면의 절대 좌표를 구합니다.
    """
    rect = RECT()
    # 클라이언트 영역의 크기 구하기
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)): return None
    
    # 클라이언트 영역의 좌상단(0,0)을 스크린 절대 좌표로 변환
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)): return None
    
    return pt.x, pt.y, rect.right, rect.bottom

class VisionSystem:
    def __init__(self):
        # 초기화 시 안전한 기본값 설정
        self.capture_area = {"top": 0, "left": 0, "width": 1366, "height": 768}
        self.window_found = False
        self.hwnd = None
        
        self.kill_roi = None 
        self.minimap_roi = None 
        
        self.last_ocr_time = 0
        self.current_kill_count = 0
        self.player_pos = (0, 0)
        
        self.skill_rois = {}
        self.skill_debug_info = {}
        # MSS 객체 재사용을 위해 여기서 생성하지 않고 with문 사용 권장
        # 하지만 빈번한 호출 시 오버헤드를 줄이기 위해 멤버로 둘 수도 있음.
        # 여기서는 안정성을 위해 매 호출마다 with mss() 사용 (mss는 가벼움)

    def find_maple_window(self):
        """메이플스토리 창을 찾아 캡처 영역을 갱신합니다."""
        try:
            windows = gw.getWindowsWithTitle('MapleStory')
            if not windows:
                return False
            
            win = windows[0]
            self.hwnd = win._hWnd
            
            # 최소화 상태면 복구
            if win.isMinimized:
                win.restore()
                time.sleep(0.5) # 복구 대기
            
            # [핵심] 진단 도구에서 검증된 좌표 계산 방식 사용
            rect = get_client_area_on_screen(self.hwnd)
            if not rect: return False
            
            x, y, w, h = rect
            # 크기가 0이면 캡처 불가
            if w <= 0 or h <= 0: return False

            # mss는 width, height를 int로 요구
            self.capture_area = {
                "top": int(y), 
                "left": int(x), 
                "width": int(w), 
                "height": int(h)
            }
            self.window_found = True
            # 디버깅용 출력
            print(f"✅ Window Found: {self.capture_area}")
            return True
        except Exception as e:
            print(f"Window Find Error: {e}")
            self.window_found = False
            return False

    def set_roi(self, rect):
        self.kill_roi = rect

    def set_minimap_roi(self, rect):
        self.minimap_roi = rect

    def get_player_position(self, frame):
        """미니맵 내 노란색 점 추적"""
        if not self.minimap_roi: return 0, 0
        x, y, w, h = self.minimap_roi
        
        # 프레임 범위 체크
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return 0, 0
            
        minimap = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
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
        """OCR 처리"""
        if not self.kill_roi: return self.current_kill_count
        if time.time() - self.last_ocr_time < 0.5: return self.current_kill_count
        
        self.last_ocr_time = time.time()
        try:
            x, y, w, h = self.kill_roi
            if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                return self.current_kill_count
                
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_not(thresh)
            
            txt = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            if txt.strip().isdigit():
                self.current_kill_count = int(txt.strip())
        except:
            pass
        return self.current_kill_count

    def capture_and_analyze(self):
        """봇용 통합 메서드"""
        # 창 정보를 못 찾았거나, 크기가 이상하면 재탐색
        if not self.window_found or self.capture_area["width"] <= 0:
            if not self.find_maple_window():
                # 실패 시 검은 화면 반환
                return np.zeros((100, 100, 3), dtype=np.uint8), 0, 0, 0, 0
        
        try:
            with mss.mss() as sct:
                # grab 시 모니터 범위를 벗어나면 mss가 에러를 낼 수 있음 -> try/except 처리
                img_np = np.array(sct.grab(self.capture_area))
            
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            
            # 분석
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            entropy = np.sum(edges) / 255
            
            kill = self.get_kill_count_ocr(frame)
            px, py = self.get_player_position(frame)
            
            return frame, entropy, kill, px, py
            
        except Exception as e:
            print(f"Capture Error: {e}")
            self.window_found = False # 다음 루프 때 다시 창을 찾도록 유도
            return np.zeros((100, 100, 3), dtype=np.uint8), 0, 0, 0, 0
        
    def set_skill_roi(self, skill_name, rect, frame=None, threshold=None):
        """
        [수정] 스킬별 맞춤형 기준값 자동 설정
        사용자가 ROI를 설정할 때(스킬이 활성화된 상태일 때)의 밝기를 측정하여,
        그보다 20%~30% 어두워지면 쿨타임으로 인식하도록 기준을 잡습니다.
        """
        if not hasattr(self, 'skill_rois'): self.skill_rois = {}

        # 기본값 (저장된 값이 없거나 화면이 없을 때)
        final_threshold = 100.0
        
        # 1. 저장된 설정 불러오기 (threshold가 직접 전달된 경우)
        if threshold is not None:
            final_threshold = float(threshold)
            
        # 2. [핵심] 화면을 보고 '현재 밝기'를 기준으로 자동 설정
        elif frame is not None:
            x, y, w, h = rect
            # 영역 안전장치
            if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                roi_img = frame[y:y+h, x:x+w]
                hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
                
                # 현재(활성화 상태)의 평균 명도(Value) 측정
                active_v = np.mean(hsv[:, :, 2])
                
                # 기준값 설정: 현재 밝기의 75% 수준으로 잡음
                # 예: 활성화(200) -> 기준(150). 쿨타임되어 100이 되면 True 반환.
                final_threshold = active_v * 0.75
                
                print(f"✅ [{skill_name}] 기준값 설정 완료: 현재밝기({active_v:.1f}) -> 기준({final_threshold:.1f})")
            
        self.skill_rois[skill_name] = {
            'rect': rect,
            'threshold': final_threshold
        }

    def scan_skill_status(self, frame):
        if not hasattr(self, 'skill_rois') or not self.skill_rois:
            return

        for name, data in self.skill_rois.items():
            x, y, w, h = data['rect']
            thresh = data['threshold']

            # 범위 체크
            if y+h > frame.shape[0] or x+w > frame.shape[1]: 
                continue

            # HSV 변환 및 밝기 측정
            roi = frame[y:y+h, x:x+w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            current_v = np.mean(hsv[:, :, 2]) # 명도(Value)

            # 쿨타임 판단 (현재 밝기 < 기준값)
            is_cool = current_v < thresh

            # 정보 저장 (GUI에서 가져다 쓸 것임)
            self.skill_debug_info[name] = {
                "val": current_v,
                "thr": thresh,
                "is_cool": is_cool
            }

    def is_skill_on_cooldown(self, skill_name, frame):
        """
        [수정] 저장된 '개별 기준값'과 비교하여 쿨타임 판단
        1순위: scan_skill_status에서 미리 계산한 값 사용 (GUI 표시와 로직 동기화)
        2순위: 정보가 없으면 직접 계산 (안전장치)
        """
        # 1. 방금 scan_skill_status()가 갱신해둔 정보가 있다면 즉시 반환
        #    (Agent와 GUI가 완전히 동일한 판단 결과를 공유하게 됨)
        if hasattr(self, 'skill_debug_info') and skill_name in self.skill_debug_info:
            return self.skill_debug_info[skill_name]["is_cool"]

        # ---------------------------------------------------------
        # 2. 정보가 없을 경우 직접 계산 (Fallback Logic)
        #    (scan_skill_status가 호출되지 않았을 때를 대비한 안전장치)
        # ---------------------------------------------------------
        
        # 예외 처리: 설정이 없거나 프레임이 비어있으면 False(준비됨) 처리
        if not hasattr(self, 'skill_rois') or skill_name not in self.skill_rois or frame is None:
            return False 
            
        data = self.skill_rois[skill_name]
        x, y, w, h = data['rect']
        stored_threshold = data['threshold'] # 이 스킬만의 고유 기준값
        
        # 이미지 범위 체크 (화면 밖으로 나가는 경우 방지)
        if y+h > frame.shape[0] or x+w > frame.shape[1]: 
            return False
        
        # ROI 추출 및 HSV 변환
        roi_img = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # 현재 화면의 명도(Value) 계산
        # 채도(S)는 무시하고, 밝기(V)만으로 판단하는 것이 가장 정확함
        current_v = np.mean(hsv[:, :, 2])
        
        # [판단] 현재 밝기가 설정해둔 기준보다 낮으면(어두우면) 쿨타임으로 간주
        is_cooldown = current_v < stored_threshold
        
        return is_cooldown