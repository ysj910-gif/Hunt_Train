# modules/brain.py
import time
import random
import config
import json

class SkillManager:
    def __init__(self):
        self.cooldowns = {} 
        self.durations = {} # [신규] 지속 시간 저장용 딕셔너리
        self.last_used = {}

    # [수정] 인자를 2개(쿨타임, 지속시간) 받도록 변경
    def update_skill_list(self, new_skill_dict, new_duration_dict):
        """
        GUI에서 설정한 쿨타임과 지속시간을 업데이트
        """
        self.cooldowns = new_skill_dict
        self.durations = new_duration_dict
        
        # 기존 사용 기록은 유지하되, 삭제된 스킬은 제거
        new_last_used = {}
        for skill in self.cooldowns:
            if skill in self.last_used:
                new_last_used[skill] = self.last_used[skill]
            else:
                new_last_used[skill] = 0.0
        self.last_used = new_last_used

    def is_ready(self, skill):
        """쿨타임이 돌았는지 확인"""
        if skill not in self.cooldowns: return True
        elapsed = time.time() - self.last_used.get(skill, 0)
        return elapsed >= self.cooldowns[skill]

    # [신규] 스킬이 현재 지속(설치) 중인지 확인
    def is_active(self, skill):
        if skill not in self.durations: return False # 지속시간 설정 안 함
        if self.durations[skill] <= 0: return False # 즉발 스킬임
        
        elapsed = time.time() - self.last_used.get(skill, 0)
        # 경과 시간이 지속 시간보다 짧으면 'Active(켜져있음)' 상태
        return elapsed < self.durations[skill]

    def use(self, skill):
        self.last_used[skill] = time.time()

    def get_remaining(self, skill):
        if skill not in self.cooldowns: return 0
        elapsed = time.time() - self.last_used.get(skill, 0)
        return max(0.0, self.cooldowns[skill] - elapsed)

class StrategyBrain:
    def __init__(self, skill_manager):
        self.sm = skill_manager
        self.footholds = [] # 발판 데이터 저장소

    # modules/brain.py (Brain 클래스 내부)

    def load_map_file(self, file_path):
        """JSON 파일에서 발판 및 스폰 정보를 읽어옵니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.footholds = []
            self.spawn_points = [] # [신규] 스폰 좌표 리스트
            
            # 1. 발판 로드 (기존 코드)
            platforms = data.get("platforms", [])
            if not platforms and isinstance(data, list):
                platforms = data

            for p in platforms:
                if "x_start" in p and "x_end" in p and "y" in p:
                    self.footholds.append((p["x_start"], p["y"], p["x_end"], p["y"]))
            
            # 2. [신규] 스폰 포인트 로드
            spawns = data.get("spawns", [])
            for s in spawns:
                if "x" in s and "y" in s:
                    self.spawn_points.append((s["x"], s["y"]))
            
            print(f"✅ 맵 로드 성공: 발판 {len(self.footholds)}개, 스폰 포인트 {len(self.spawn_points)}개")
            return True
            
        except Exception as e:
            print(f"❌ 맵 로드 실패: {e}")
            self.footholds = []
            self.spawn_points = []
            return False
        
    def analyze_spawn_points(self):
        """스폰 포인트들의 분포를 분석하여 '설치기 명당'을 찾습니다."""
        if not self.spawn_points: return

        # 1. 모든 스폰 포인트의 무게중심(Centroid) 계산
        sum_x = sum(p[0] for p in self.spawn_points)
        sum_y = sum(p[1] for p in self.spawn_points)
        center_x = sum_x / len(self.spawn_points)
        center_y = sum_y / len(self.spawn_points)

        # 2. 각 포인트별로 '외딴 정도(Isolation Score)' 계산
        # (중심에서 멀거나, 주변에 다른 스폰 포인트가 없을수록 점수가 높음)
        for i, p in enumerate(self.spawn_points):
            dist_from_center = ((p[0] - center_x)**2 + (p[1] - center_y)**2) ** 0.5
            
            # 3. 전략 설정: 중심에서 멀면 'Install', 가까우면 'Attack' 권장
            # (예: 맵 크기에 따라 기준값 300픽셀 설정)
            strategy = "Install" if dist_from_center > 300 else "Main_Hunt"
            
            # 정보 업데이트 (기존 좌표에 전략 태그 추가)
            self.spawn_points[i] = (p[0], p[1], strategy)
            
        print("✅ 지형 분석 완료: 설치기 추천 구역 설정됨")

    def decide_action(self, entropy):
        return "patrol"