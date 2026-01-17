import cv2
import mss
import numpy as np
import config

class VisionEngine:
    def __init__(self):
        self.sct = mss.mss()

    def capture_minimap(self):
        """미니맵 ROI만 정밀하게 캡처"""
        mini_img = np.array(self.sct.grab(config.MINIMAP_ROI))
        return cv2.cvtColor(mini_img, cv2.COLOR_BGRA2BGR)

    def get_character_mask(self, mini_bgr):
        """캐릭터(노란 점)만 남긴 마스크 이미지 생성 (효율 극대화 옵션)"""
        hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.YELLOW_HSV_LOWER, config.YELLOW_HSV_UPPER)
        return mask # 흑백(0, 255) 이미지로 저장됨