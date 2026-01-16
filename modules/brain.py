# modules/brain.py
import time
import random
import config
import json

class SkillManager:
    def __init__(self):
        self.cooldowns = {} # 초기에는 빈 상태 (UI에서 로드됨)
        self.last_used = {}

    def update_skill_list(self, new_skill_dict):
        """
        GUI에서 설정한 스킬 목록을 통째로 덮어씌우는 함수
        new_skill_dict 예시: {"Genesis": 30.0, "Heal": 5.0}
        """
        self.cooldowns = new_skill_dict
        
        # 기존 사용 기록은 유지하되, 삭제된 스킬은 제거
        new_last_used = {}
        for skill in self.cooldowns:
            if skill in self.last_used:
                new_last_used[skill] = self.last_used[skill]
            else:
                new_last_used[skill] = 0.0
        self.last_used = new_last_used

    def is_ready(self, skill):
        if skill not in self.cooldowns: return True
        elapsed = time.time() - self.last_used.get(skill, 0)
        return elapsed >= self.cooldowns[skill]

    def use(self, skill):
        self.last_used[skill] = time.time()

    def get_remaining(self, skill):
        if skill not in self.cooldowns: return 0
        elapsed = time.time() - self.last_used.get(skill, 0)
        return max(0.0, self.cooldowns[skill] - elapsed)

    def load_map_file(self, file_path):
        """JSON 파일에서 발판 정보를 읽어옵니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.footholds = []
            
            # 1. "platforms" 키가 있는 경우 (제공해주신 데이터 포맷)
            # 포맷: {"y": 112, "x_start": 22, "x_end": 67, ...}
            platforms = data.get("platforms", [])
            
            # 만약 "platforms" 키가 없고 리스트 자체가 데이터라면
            if not platforms and isinstance(data, list):
                platforms = data

            for p in platforms:
                # 필수 키가 존재하는지 확인
                if "x_start" in p and "x_end" in p and "y" in p:
                    x1 = p["x_start"]
                    x2 = p["x_end"]
                    y = p["y"]
                    # 그리기 쉽도록 (x1, y1, x2, y2) 형태로 저장
                    self.footholds.append((x1, y, x2, y))
            
            print(f"✅ 맵 로드 성공: 발판 {len(self.footholds)}개")
            return True
        except Exception as e:
            print(f"❌ 맵 로드 실패: {e}")
            self.footholds = []
            return False

    def decide_action(self, entropy):
        return "patrol"