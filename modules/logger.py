# modules/logger.py
import csv
import time
import os
from datetime import datetime

class DataLogger:
    def __init__(self, job_name, is_bot=False):
        # 데이터 저장 폴더 생성
        if not os.path.exists("data"):
            os.makedirs("data")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "Bot" if is_bot else "Human"
        
        self.filepath = f"data/{prefix}_{job_name}_{timestamp}.csv"
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        
        self.headers = [
            "timestamp", 
            "entropy", 
            "ult_ready", 
            "sub_ready", 
            "action_name",    
            "key_pressed",    
            "player_x",       
            "player_y",       
            "platform_id",    
            "kill_count", 
            "kill_reward",
            # [신규] 벽까지의 거리 (학습 효과 극대화)
            "dist_left",
            "dist_right"
        ]
        self.writer.writerow(self.headers)
            
        self.last_kills = 0
        print(f"✅ 데이터 수집 시작: {self.filepath}")

    def log_step(self, entropy, skill_manager, action_name, key_pressed, px, py, pid, current_kills, dist_left, dist_right):
        """
        [수정] dist_left, dist_right 인자 추가
        """
        # 보상 계산
        if current_kills < self.last_kills:
             self.last_kills = current_kills
        reward = max(0, current_kills - self.last_kills)
        
        ult_ready = 1 if skill_manager.is_ready("ultimate") else 0
        sub_ready = 1 if skill_manager.is_ready("sub_attack") else 0
        
        row = [
            time.time(),
            f"{entropy:.2f}",
            ult_ready,
            sub_ready,
            action_name,
            key_pressed,
            px,               
            py,               
            pid,      
            current_kills,
            reward,
            # [신규] 거리 데이터 저장
            round(dist_left, 1),
            round(dist_right, 1)
        ]
        
        self.writer.writerow(row)
        self.last_kills = current_kills

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print("✅ 로그 파일 저장 완료")