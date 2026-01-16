# modules/brain.py
import time
import random
import config
import json

class SkillManager:
    def __init__(self):
        self.cooldowns = config.DEFAULT_COOLDOWNS.copy()
        self.last_used = {k: 0.0 for k in self.cooldowns}

    def set_cooldown(self, skill, seconds):
        self.cooldowns[skill] = float(seconds)

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

class StrategyBrain:
    def __init__(self, skill_manager):
        self.sm = skill_manager
        self.threshold = 3000
        self.current_map_data = None # 맵 데이터 저장용

    def load_map_file(self, file_path):
        """JSON 맵 파일을 불러오는 기능 추가"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.current_map_data = json.load(f)
            print(f"✅ 맵 로드 성공: {file_path}")
            return True
        except Exception as e:
            print(f"❌ 맵 로드 실패: {e}")
            return False

    def decide_action(self, entropy):
        """엔트로피 수치를 보고 행동 결정"""
        
        # 1. 광역기 (몹이 매우 많음)
        if entropy > self.threshold * 1.5 and self.sm.is_ready("ultimate"):
            return "ultimate"
        
        # 2. 서브 스킬 (몹이 적당함)
        elif entropy > self.threshold and self.sm.is_ready("sub_attack"):
            return "sub_attack"
        
        # 3. 평타 (몹이 조금 있음)
        elif entropy > self.threshold * 0.8:
            return "attack"
        
        # 4. 몬스터 없음 -> 이동
        else:
            return "patrol"