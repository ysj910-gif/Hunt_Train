# modules/brain.py
import time
import random
import config
import json
import math

class SkillManager:
    def __init__(self):
        self.cooldowns = {} 
        self.durations = {} 
        self.last_used = {}

    def update_skill_list(self, new_skill_dict, new_duration_dict):
        """GUI에서 설정한 쿨타임과 지속시간을 업데이트"""
        self.cooldowns = new_skill_dict
        self.durations = new_duration_dict
        
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

    def is_active(self, skill):
        """스킬이 현재 지속(설치) 중인지 확인"""
        if skill not in self.durations: return False 
        if self.durations[skill] <= 0: return False 
        
        elapsed = time.time() - self.last_used.get(skill, 0)
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
        self.footholds = [] 
        self.spawn_points = [] 
        self.install_spots = [] # 설치기 명당 목록

    def load_map_file(self, file_path):
        """JSON 파일에서 발판 및 스폰 정보를 읽어옵니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.footholds = []
            self.spawn_points = []
            
            # 1. 발판 로드
            platforms = data.get("platforms", [])
            for p in platforms:
                if "x_start" in p and "x_end" in p and "y" in p:
                    self.footholds.append((p["x_start"], p["y"], p["x_end"], p["y"]))
            
            # 2. 스폰 포인트 로드
            spawns = data.get("spawns", [])
            for s in spawns:
                if "x" in s and "y" in s:
                    # 딕셔너리 형태로 저장하여 관리 용이하게 함
                    self.spawn_points.append({'x': s["x"], 'y': s["y"], 'desc': s.get('desc', '')})
            
            print(f"✅ 맵 로드 성공: 발판 {len(self.footholds)}개, 스폰 {len(self.spawn_points)}개")
            
            # 맵 로드 직후 분석 수행
            self.analyze_spawn_points()
            
            return True
            
        except Exception as e:
            print(f"❌ 맵 로드 실패: {e}")
            self.footholds = []
            self.spawn_points = []
            return False
        
    def analyze_spawn_points(self):
        """
        스폰 포인트 분석: 
        상하 이동(Y축)이 좌우 이동(X축)보다 어렵다는 점을 반영하여
        '체감 거리'가 중심에서 먼 곳을 설치기 명당으로 선정합니다.
        """
        if not self.spawn_points: return

        # 1. X축, Y축 각각의 무게중심(Centroid) 계산
        x_coords = [p['x'] for p in self.spawn_points]
        y_coords = [p['y'] for p in self.spawn_points]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        # 2. 이동 비용 가중치 설정 (Y축 페널티)
        X_WEIGHT = 1.0
        Y_WEIGHT = 2.5 

        self.install_spots = [] # 초기화

        # 3. 각 포인트별 '체감 격리도(Isolation Score)' 계산
        for p in self.spawn_points:
            dx = abs(p['x'] - center_x)
            dy = abs(p['y'] - center_y)
            
            # [가중치 적용 거리 공식]
            weighted_dist = math.sqrt((dx * X_WEIGHT)**2 + (dy * Y_WEIGHT)**2)
            
            # 4. 전략 설정: 가중치 거리가 일정 이상이면 'Install' 구역으로 분류
            # 기준값(Threshold)은 맵 크기에 따라 다르지만, 보통 300~400 정도면 외곽으로 간주
            if weighted_dist > 350:
                p['strategy'] = "Install"
                p['score'] = weighted_dist
                self.install_spots.append(p)
            else:
                p['strategy'] = "Main_Hunt"
                p['score'] = weighted_dist
            
        # 점수가 높은(더 외진) 순서대로 정렬
        self.install_spots.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✅ 지형 분석 완료 (Y가중치 {Y_WEIGHT}): 설치기 명당 {len(self.install_spots)}곳 선정")
        for i, spot in enumerate(self.install_spots):
            print(f"   📍 명당 {i+1}: ({spot['x']}, {spot['y']}) - Score: {spot['score']:.1f}")

    def decide_action(self, entropy, player_x, player_y):
        """
        현재 상태를 보고 행동을 결정 (예시 로직)
        """
        # 설치기가 쿨타임이 찼고, 설치기 명당 근처에 있다면?
        # (이 부분은 나중에 구체적인 스킬 사용 로직과 연동해야 합니다)
        
        return "patrol"