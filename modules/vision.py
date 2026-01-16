# modules/vision.py
import mss
import cv2
import numpy as np
import pygetwindow as gw # (창 찾기 기능 포함)
import config

class VisionSystem:
    def __init__(self):
        # self.sct = mss.mss()  <-- [삭제] 여기서 미리 만들면 스레드 충돌 남
        self.capture_area = config.DEFAULT_CAPTURE_AREA
        self.window_found = False

    def find_maple_window(self):
        """제목이 'MapleStory'인 창을 찾아 위치를 자동 설정"""
        try:
            windows = gw.getWindowsWithTitle('MapleStory')
            if not windows:
                print("❌ 'MapleStory' 창을 찾을 수 없습니다.")
                return False
                
            win = windows[0]
            if win.isMinimized:
                win.restore()
            
            # 윈도우 좌표 업데이트
            self.capture_area = {
                "top": win.top + 30,
                "left": win.left + 8,
                "width": win.width - 16,
                "height": win.height - 38
            }
            self.window_found = True
            print(f"✅ 메이플 창 발견: {self.capture_area}")
            return True
        except Exception as e:
            print(f"⚠️ 창 찾기 오류: {e}")
            return False

    def capture_and_analyze(self):
        # 창 위치를 못 찾았으면 한번 시도
        if not self.window_found:
            self.find_maple_window()
        
        try:
            # [수정] 스레드 충돌 방지를 위해 캡처 시마다 with 구문 사용
            with mss.mss() as sct:
                img_np = np.array(sct.grab(self.capture_area))
                
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            entropy_score = np.sum(edges) / 255
            
            return frame, entropy_score
        except Exception as e:
            # 캡처 실패 시 (창이 꺼지거나 최소화됨) 빈 화면 반환
            # print(f"캡처 오류: {e}") # 너무 자주 뜨면 주석 처리
            return np.zeros((100, 100, 3), dtype=np.uint8), 0