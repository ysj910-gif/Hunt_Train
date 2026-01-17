# modules/logger.py
import csv
import time
import os
from datetime import datetime

class DataLogger:
    def __init__(self, job_name): # [수정] job_name을 받도록 변경
        # 데이터 저장 폴더 생성
        if not os.path.exists("data"):
            os.makedirs("data")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 파일명에 직업 이름 포함
        self.filepath = f"data/{job_name}_{timestamp}.csv"
        
        # [핵심] 파일을 미리 열어둡니다 (속도 향상 및 멈춤 방지)
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        
        # [수정] train.py 학습에 필요한 컬럼명(player_x 등)으로 통일
        self.headers = [
            "timestamp", 
            "entropy", 
            "ult_ready", 
            "sub_ready", 
            "action_name",    # 스킬 이름 (예: Attack)
            "key_pressed",    # 실제 눌린 키 (예: ctrl) - train.py의 target
            "player_x",       # [추가] 좌표 X
            "player_y",       # [추가] 좌표 Y
            "platform_id",    # [추가] 발판 ID (나중에 gui.py에서 계산 필요)
            "kill_count", 
            "kill_reward"
        ]
        self.writer.writerow(self.headers)
            
        self.last_kills = 0
        print(f"✅ 고속 데이터 수집 시작: {self.filepath}")

    # [수정] gui.py가 보내주는 7개 인자를 모두 받도록 수정 (+ platform_id는 선택)
    def log_step(self, entropy, skill_manager, action_name, key_pressed, px, py, current_kills, platform_id=-1):
        """
        한 번의 행동(프레임)마다 데이터를 저장함
        """
        # 보상 계산
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
            action_name,
            key_pressed,
            px,               # player_x 저장
            py,               # player_y 저장
            platform_id,      # platform_id 저장
            current_kills,
            reward
        ]
        
        # 열어둔 파일에 즉시 기록 (I/O 지연 없음)
        self.writer.writerow(row)
        
        self.last_kills = current_kills

    def close(self):
        """녹화 종료 시 호출하여 파일 안전하게 닫기"""
        if self.file:
            self.file.close()
            self.file = None
            print("✅ 로그 파일 저장 완료 및 닫기")