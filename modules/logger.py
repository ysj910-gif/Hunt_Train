# modules/logger.py
import csv
import time
import os
from datetime import datetime

class DataLogger:
    def __init__(self, job_name="Unknown"):
        # 데이터 저장 폴더 생성
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # 파일명: data/Bishop_20231025_1230.csv (직업명 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_job = "".join(x for x in job_name if x.isalnum()) # 특수문자 제거
        if not clean_job: clean_job = "NoName"
            
        self.filepath = f"data/{clean_job}_{timestamp}.csv"
        
        # CSV 헤더 (action -> skill_name으로 변경)
        self.headers = [
            "timestamp", 
            "entropy",       
            "ult_ready",     
            "sub_ready",     
            "skill_name",    # 변경됨: 사용한 스킬 이름
            "key_pressed",   # 참고용: 실제 누른 키
            "kill_count",    
            "kill_reward"    
        ]
        
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
        self.last_kills = 0
        print(f"✅ 데이터 수집 시작: {self.filepath}")

    def log_step(self, entropy, skill_manager, skill_name, key_char, current_kills):
        """
        한 번의 행동(프레임)마다 데이터를 저장함
        """
        # 보상 계산
        reward = max(0, current_kills - self.last_kills)
        
        # 쿨타임 상태 (Role 기반 확인)
        ult_ready = 1 if skill_manager.is_ready("ultimate") else 0
        sub_ready = 1 if skill_manager.is_ready("sub_attack") else 0
        
        # 데이터 행 생성
        row = [
            time.time(),
            f"{entropy:.2f}",
            ult_ready,
            sub_ready,
            skill_name,   # Genesis, Teleport 등
            key_char,     # r, e, q 등
            current_kills,
            reward
        ]
        
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        self.last_kills = current_kills