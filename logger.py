# modules/logger.py
import csv
import time
import os
from datetime import datetime

class DataLogger:
    def __init__(self):
        # 데이터 저장 폴더 생성
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # 파일명: data/play_log_20231025_1230.csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"data/play_log_{timestamp}.csv"
        
        # CSV 헤더 작성 (상태, 행동, 결과)
        self.headers = [
            "timestamp", 
            "entropy",       # 상태: 화면 복잡도
            "ult_ready",     # 상태: 광역기 사용 가능 여부 (1/0)
            "sub_ready",     # 상태: 서브기 사용 가능 여부 (1/0)
            "user_action",   # 행동: 사용자가 누른 키
            "kill_count",    # 결과: 현재 킬 카운트
            "kill_reward"    # 보상: 직전 대비 잡은 몬스터 수
        ]
        
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
        self.last_kills = 0
        print(f"✅ 데이터 수집 시작: {self.filepath}")

    def log_step(self, entropy, skill_manager, action, current_kills):
        """
        한 번의 행동(프레임)마다 데이터를 저장함
        """
        # 보상 계산 (이번 행동으로 몇 마리 잡았는지)
        # OCR 특성상 숫자가 튀는 경우(인식 오류)를 대비해 음수는 0 처리
        reward = max(0, current_kills - self.last_kills)
        
        # 쿨타임 상태 확인
        ult_ready = 1 if skill_manager.is_ready("ultimate") else 0
        sub_ready = 1 if skill_manager.is_ready("sub_attack") else 0
        
        # 데이터 행 생성
        row = [
            time.time(),
            f"{entropy:.2f}",
            ult_ready,
            sub_ready,
            action,
            current_kills,
            reward
        ]
        
        # 파일에 쓰기
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        self.last_kills = current_kills