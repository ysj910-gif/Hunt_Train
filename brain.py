import json
import os
import random

class SkillManager:
    """스킬 쿨타임 및 사용 가능 여부 관리"""
    def __init__(self):
        self.cooldowns = {}
        self.last_used = {}
        self.durations = {} # 지속시간 관리 (버프 등)

    def update_skill_list(self, cd_dict, dur_dict=None):
        self.cooldowns = cd_dict
        if dur_dict: self.durations = dur_dict
        
        # 새로운 스킬은 last_used 0으로 초기화
        for s in cd_dict:
            if s not in self.last_used: self.last_used[s] = 0

    def is_ready(self, skill_name):
        if skill_name not in self.cooldowns: return True
        # 쿨타임 체크
        elapsed = self.get_elapsed(skill_name)
        if elapsed < self.cooldowns[skill_name]: return False
        
        # [신규] 이미 사용 중(지속시간 내)이면 준비 안 된 것으로 간주 (중복 사용 방지)
        if skill_name in self.durations:
            if elapsed < self.durations[skill_name]: return False
            
        return True

    def is_active(self, skill_name):
        """현재 스킬(버프/설치기)이 지속시간 중인지 확인"""
        if skill_name not in self.durations: return False
        return self.get_elapsed(skill_name) < self.durations[skill_name]

    def use(self, skill_name):
        import time
        self.last_used[skill_name] = time.time()

    def get_elapsed(self, skill_name):
        import time
        return time.time() - self.last_used.get(skill_name, 0)
        
    def get_remaining(self, skill_name):
        if skill_name not in self.cooldowns: return 0
        rem = self.cooldowns[skill_name] - self.get_elapsed(skill_name)
        return max(0, rem)


class StrategyBrain:
    """맵 정보 관리 및 전략 판단"""
    def __init__(self, skill_manager):
        self.sm = skill_manager
        self.footholds = [] # (x1, y1, x2, y2) 형태로 저장

    def load_map_file(self, path):
        if not os.path.exists(path): return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.footholds = []
            
            # [수정] Royal Library 형식 ("platforms": [{"x_start":...}]) 지원
            if 'platforms' in data:
                for p in data['platforms']:
                    if isinstance(p, dict):
                        # 딕셔너리 형태 파싱
                        x1 = p.get('x_start', 0)
                        x2 = p.get('x_end', 0)
                        y = p.get('y', 0)
                        # y1=y2=y (평평한 발판 가정)
                        self.footholds.append((x1, y, x2, y))
                    elif isinstance(p, list) and len(p) >= 4:
                        # 리스트 형태 파싱
                        self.footholds.append(tuple(p[:4]))

            # 기존 형식 ("footholds": [[x1,y1,x2,y2]...]) 지원
            elif 'footholds' in data:
                for p in data['footholds']:
                    if len(p) >= 4:
                        self.footholds.append(tuple(p[:4]))
            
            print(f"✅ 맵 로드 성공: 발판 {len(self.footholds)}개")
            return True
            
        except Exception as e:
            print(f"❌ 맵 파일 로드 오류: {e}")
            return False

    def get_platform_id(self, px, py):
        # 가장 가까운 발판 찾기 (단순 구현)
        best_id = -1
        min_dist = 50 # 50px 이내만 인정
        
        for i, (x1, y1, x2, y2) in enumerate(self.footholds):
            # X 범위 안에 있는지
            if x1 <= px <= x2:
                dist = abs(py - y1)
                if dist < min_dist:
                    min_dist = dist
                    best_id = i
        return best_id