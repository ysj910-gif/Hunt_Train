import numpy as np
import time
import random
import json
import heapq

class InstallSkill:
    """설치기 정보 정의"""
    def __init__(self, name, up, down, left, right, duration):
        self.name = name
        self.real_range = {'up': up, 'down': down, 'left': left, 'right': right}
        self.duration = duration

class PatrolPlanner:
    def __init__(self):
        self.spawn_points = []
        self.active_installs = []   # 현재 맵에 깔려있는 스킬들
        self.current_target = None
        self.map_floor_y = 100
        
        self.SCALE_RATIO = 0.055 
        self.VISIT_THRESHOLD = 6.0 
        
        # [수정] 단일 스킬 -> 스킬 리스트
        self.install_skills = [] 
        self.next_skill_to_use = None # 다음에 사용할 스킬
        self.current_installing_skill = None

    def load_map(self, map_json_path):
        try:
            with open(map_json_path, 'r') as f:
                data = json.load(f)
            
            platforms = data.get('platforms', [])
            avg_plat_y = 0
            if platforms:
                ys = [p['y'] for p in platforms]
                self.map_floor_y = max(ys)
                avg_plat_y = sum(ys) / len(ys)
            
            raw_spawns = []
            if 'spawns' in data:
                raw_spawns = [(s['x'], s['y']) for s in data['spawns']]
            else:
                for key in data:
                    if isinstance(data[key], list):
                        for item in data[key]:
                            if isinstance(item, dict) and item.get('desc') == 'Auto Spawn':
                                raw_spawns.append((item['x'], item['y']))
            
            self.spawn_points = []
            if raw_spawns and avg_plat_y > 0:
                avg_spawn_y = sum(s[1] for s in raw_spawns) / len(raw_spawns)
                diff = avg_spawn_y - avg_plat_y
                if diff > 20: 
                    for (x, y) in raw_spawns:
                        self.spawn_points.append((x, min(int(y - diff), self.map_floor_y)))
                else:
                    self.spawn_points = raw_spawns
            else:
                self.spawn_points = raw_spawns
                
            print(f"🗺️ [Patrol] 스폰 포인트 {len(self.spawn_points)}개 로드 완료")
            print(f"   - 등록된 설치기 개수: {len(self.install_skills)}개")

        except Exception as e:
            print(f"Error loading map: {e}")

    def _is_covered(self, point):
        px, py = point
        now = time.time()
        # 만료된 설치기 제거
        self.active_installs = [ins for ins in self.active_installs if ins['expiry'] > now]
        
        for ins in self.active_installs:
            ix, iy = ins['pos']
            skill = ins['skill']
            
            up = skill.real_range['up'] * self.SCALE_RATIO
            down = skill.real_range['down'] * self.SCALE_RATIO
            left = skill.real_range['left'] * self.SCALE_RATIO
            right = skill.real_range['right'] * self.SCALE_RATIO
            
            if (ix - left <= px <= ix + right) and (iy - up <= py <= iy + down):
                return True
        return False

    def get_next_skill(self):
        """
        사용 가능한(아직 설치 안 된) 스킬을 찾아서 반환
        단순하게 이름으로 구분 (같은 이름의 스킬을 여러 개 등록했으면 여러 번 사용 가능)
        """
        now = time.time()
        # 현재 활성화된 스킬들의 이름 목록
        active_names = [ins['skill'].name for ins in self.active_installs if ins['expiry'] > now]
        
        # 등록된 스킬 중 활성화되지 않은 첫 번째 스킬 반환
        # (예: 파운틴, 야누스1, 야누스2 순서로 등록되어 있다면 순서대로 체크)
        # 주의: 동일한 스킬을 여러 번 쓰고 싶으면 GUI에 여러 줄로 등록해야 함 (예: 야누스, 야누스)
        
        # 간단한 로직: 활성화된 개수 < 등록된 개수 체크
        # 하지만 특정 스킬 매칭이 필요하므로, 여기서는 "사용 안 된 객체"를 찾음
        
        # active_installs에 있는 skill 객체 자체를 비교
        active_objs = [ins['skill'] for ins in self.active_installs if ins['expiry'] > now]
        
        for skill in self.install_skills:
            if skill not in active_objs:
                return skill
        
        return None # 모든 스킬이 쿨타임(지속시간) 중

    def get_optimum_target(self, player_x, player_y, install_ready=False):
        # 1. 만료된 설치기 정리 (가장 먼저 수행)
        now = time.time()
        self.active_installs = [ins for ins in self.active_installs if ins['expiry'] > now]

        # 2. 커버되지 않은(사냥해야 할) 포인트들 추출
        # (_is_covered는 스킬의 사각형 범위만 체크함)
        uncovered_points = [p for p in self.spawn_points if not self._is_covered(p)]
        
        if not uncovered_points:
            return (player_x, player_y), "All Covered"

        # ---------------------------------------------------------
        # [모드 1] 설치기 설치 (Install Mode)
        # ---------------------------------------------------------
        next_skill = self.get_next_skill()
        
        if install_ready and next_skill:
            self.next_skill_to_use = next_skill 
            
            best_score = -1
            best_target = uncovered_points[0]
            
            range_w = (next_skill.real_range['left'] + next_skill.real_range['right']) * self.SCALE_RATIO
            
            for cand in uncovered_points:
                # 설치기 주변에 적이 얼마나 많은지 체크 (설치 효율 계산)
                count = 0
                for other in uncovered_points:
                    if abs(other[0] - cand[0]) < range_w: 
                        count += 1
                
                # [추가] 이미 설치된 다른 설치기와 너무 가까우면 설치 후보에서 제외 (중복 설치 방지)
                too_close = False
                for ins in self.active_installs:
                    ix, iy = ins['pos']
                    if np.hypot(cand[0]-ix, cand[1]-iy) < 150: # 150px 이내면 너무 가까움
                        too_close = True; break
                
                if too_close: continue # 스킵

                if count > best_score:
                    best_score = count
                    best_target = cand
            
            self.current_target = best_target
            return best_target, "Install Position"
            
        # ---------------------------------------------------------
        # [모드 2] 일반 순찰 (Patrol Mode) - 여기가 중요!
        # ---------------------------------------------------------
        else:
            best_target = None
            min_score = float('inf') # 점수가 낮을수록 좋음 (거리 기반 Cost)

            for p in uncovered_points:
                # A. 기본 점수: 플레이어와의 거리 (가까울수록 좋음)
                dist = np.hypot(p[0]-player_x, p[1]-player_y)
                
                # 너무 가까운 포인트(이미 도착한 곳)는 무시
                if dist <= self.VISIT_THRESHOLD: 
                    continue
                
                score = dist 

                # B. [핵심] 회피 로직 (Repulsion Logic)
                # 활성화된 설치기 위치 주변에는 페널티를 부여해 봇이 안 가도록 만듦
                for ins in self.active_installs:
                    ix, iy = ins['pos']
                    # 설치기와의 직선 거리 계산
                    dist_to_install = np.hypot(p[0]-ix, p[1]-iy)
                    
                    # 설치기 반경 200px 이내의 포인트는 점수를 폭발적으로 높임 (기피 대상)
                    if dist_to_install < 10: 
                        score += 5000.0 # 절대 선택되지 않도록 강력한 페널티
                
                # 가장 점수가 낮은(가깝고 + 설치기 없는) 곳 선택
                if score < min_score:
                    min_score = score
                    best_target = p
            
            if best_target:
                self.current_target = best_target
                return self.current_target, "Patrol Uncovered"
            else:
                # 갈 곳이 없으면(모두 설치기 근처거나 완료됨) 제자리 대기
                return (player_x, player_y), "Patrol Done (Wait)"
            
    def notify_install_used(self, px, py):
        if self.next_skill_to_use:
            skill = self.next_skill_to_use
            self.active_installs.append({
                'pos': (px, py),
                'skill': skill,
                'expiry': time.time() + skill.duration
            })
            print(f"📍 설치기({skill.name}) 등록 @ ({px:.0f}, {py:.0f}) | 지속: {skill.duration}s")
            self.next_skill_to_use = None # 초기화

class TacticalNavigator:
    def __init__(self, platform_manager, physics_model=None):
        self.pm = platform_manager
        self.patrol = PatrolPlanner()
    
    def build_graph(self, map_path=None):
        target_path = map_path if map_path else getattr(self.pm, 'map_file', None)
        if target_path: self.patrol.load_map(target_path)

    def update_combat_stats(self, px, py, kill_count): pass

    def get_move_decision(self, px, py, install_ready=False):
        if not self.patrol.spawn_points: return "None", "No Map Data"

        target_pos, mode = self.patrol.get_optimum_target(px, py, install_ready)
        tx, ty = target_pos
        
        # Floor Clamp
        floor_y = self.patrol.map_floor_y
        if ty > floor_y: ty = floor_y
        
        dx = tx - px
        dy = ty - py 
        dist = abs(dx)
        
        vertical_limit = 25 if mode == "Install Position" else 10 # 설치 시 수직 판정 더 관대하게
        
        if dist <= self.patrol.VISIT_THRESHOLD and abs(dy) < vertical_limit:
            if mode == "Install Position":
                return "None", "Positioned for Install"
            else:
                return "None", "Reached Point"

        if dy > 25:
            if py < floor_y - 5: 
                if abs(dx) < 20: return "down+jump", f"Down to {mode}"
        elif dy < -15: 
            if abs(dx) < 20: return "up+jump", f"Up to {mode}"

        action = "right" if dx > 0 else "left"
        return action, f"{mode}"

    def notify_install_success(self):
        if self.patrol.current_target:
            self.patrol.notify_install_used(self.patrol.current_target[0], self.patrol.current_target[1])
            
    def patrol_mode(self, px, py):
        act, _ = self.get_move_decision(px, py)
        return act