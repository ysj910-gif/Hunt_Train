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
        
        # [헤더 설정]
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
            "dist_left",   
            "dist_right",  
            "job_class"    
        ]
        self.writer.writerow(self.headers)
            
        self.last_kills = 0
        print(f"✅ 데이터 수집 시작: {self.filepath}")

    def log_step(self, entropy, skill_manager, action_name, key_pressed, px, py, pid, current_kills, dist_left, dist_right, job_class, key_map=None):        
        """
        [수정] key_pressed가 '세트(Set)'로 들어오면, key_map을 이용해 '명령어'로 변환합니다.
        """
        # 1. 키 입력 변환 로직 (Human Input 처리)
        final_command = "None"
        
        if isinstance(key_pressed, set): # 사람이 누른 키 집합인 경우
            if not key_pressed:
                final_command = "None"
            else:
                final_command = self._translate_keys(key_pressed, key_map)
        else:
            # 봇이 보낸 문자열(String)인 경우 그대로 사용
            final_command = str(key_pressed)

        # 2. 보상 및 쿨타임 계산
        if current_kills < self.last_kills:
             self.last_kills = current_kills
        reward = max(0, current_kills - self.last_kills)
        
        ult_ready = 1 if skill_manager.is_ready("ultimate") else 0
        sub_ready = 1 if skill_manager.is_ready("sub_attack") else 0
        
        # 3. CSV 기록
        row = [
            time.time(),
            f"{entropy:.2f}",
            ult_ready,
            sub_ready,
            action_name,
            final_command, # 변환된 명령어가 기록됨
            px,               
            py,               
            pid,      
            current_kills,
            reward,
            round(dist_left, 1),
            round(dist_right, 1),
            job_class
        ]
        
        self.writer.writerow(row)
        self.last_kills = current_kills

    def _translate_keys(self, keys, key_map):
        """
        물리 키 집합({'left', 'e', 'r'})을 의미 있는 명령("Left+DoubleJump+Attack")으로 변환
        """
        cmd_parts = []
        
        # 1. 점프 키 식별 (key_map에서 'Jump'라는 이름이 포함된 키 찾기)
        jump_keys = ['space', 'alt'] # 기본 점프 키 후보
        if key_map:
            # 사용자가 설정한 스킬 중 이름에 'jump'가 들어가는 키 찾기
            for name, key in key_map.items():
                if 'jump' in name.lower():
                    jump_keys.append(key)
        
        # 실제 눌린 키 중 점프 키가 있는지 확인
        is_jumping = any(k in jump_keys for k in keys)
        
        # 2. 방향키 처리
        if 'left' in keys: cmd_parts.append('left')
        elif 'right' in keys: cmd_parts.append('right')
        elif 'up' in keys: cmd_parts.append('up')
        elif 'down' in keys: cmd_parts.append('down')
        
        # 3. 점프/더블점프 변환
        if is_jumping:
            if any(k in keys for k in ['left', 'right']):
                cmd_parts.append('double_jump') # 이동 + 점프 = 플점
            else:
                cmd_parts.append('jump') # 제자리 점프
        
        # 4. 스킬/공격 키 변환 (매핑된 이름 사용)
        if key_map:
            for name, key in key_map.items():
                # 점프는 위에서 처리했으니 제외
                if 'jump' in name.lower(): continue
                
                if key in keys:
                    cmd_parts.append(name) # 예: 'r' -> 'MainAttack'
        
        # 매핑 안 된 키들 중 특수 키 추가 (Shift, Ctrl 등)
        for k in keys:
            if k in ['shift', 'ctrl', 'alt', 'z', 'x', 'c'] and k not in jump_keys:
                # 이미 스킬명으로 추가된 키가 아니면 추가
                if key_map and k in key_map.values(): continue
                cmd_parts.append(k)

        if not cmd_parts: return "None"
        return "+".join(cmd_parts)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print("✅ 로그 파일 저장 완료")