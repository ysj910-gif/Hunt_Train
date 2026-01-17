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
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_job = "".join(x for x in job_name if x.isalnum())
        if not clean_job: clean_job = "NoName"
            
        self.filepath = f"data/{clean_job}_{timestamp}.csv"
        
        # [수정] CSV 헤더에 좌표(player_x, player_y) 추가
        self.headers = [
            "timestamp", 
            "entropy",       
            "skill_name",    
            "key_pressed",   
            "active_skills", 
            "player_x",      # [신규]
            "player_y",      # [신규]
            "kill_count",    
            "kill_reward"    
        ]
        
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
        self.last_kills = 0
        print(f"✅ 데이터 수집 시작: {self.filepath}")

    # [수정] 인자에 px, py 추가
    def log_step(self, entropy, skill_manager, skill_name, key_char, px, py, current_kills):
        reward = max(0, current_kills - self.last_kills)
        
        # 활성 스킬(설치기) 목록 만들기
        active_list = []
        # skill_manager에 durations 속성이 있는지 확인 (안전장치)
        if hasattr(skill_manager, 'durations'): 
            for s_name in skill_manager.durations:
                if skill_manager.is_active(s_name):
                    active_list.append(s_name)
        
        active_str = "|".join(active_list) if active_list else "None"
        
        # [수정] 좌표 데이터 포함하여 저장
        row = [
            time.time(),
            f"{entropy:.2f}",
            skill_name,
            key_char,
            active_str,
            px, # [신규] 좌표 X
            py, # [신규] 좌표 Y
            current_kills,
            reward
        ]
        
        try:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"로그 저장 실패: {e}")
            
        self.last_kills = current_kills