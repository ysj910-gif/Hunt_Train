import math
import time
import heapq
from collections import deque, defaultdict

# A* 경로 탐색에 사용할 노드 클래스
class PathNode:
    def __init__(self, x, y, g, h, parent=None, action=None):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.action = action
    
    def __lt__(self, other):
        return self.f < other.f

# 그래프상의 지점 (웨이포인트) 클래스
class Waypoint:
    def __init__(self, x, y, platform_id):
        self.x = x
        self.y = y
        self.pid = platform_id

class TacticalNavigator:
    def __init__(self, platform_manager, physics_learner):
        self.pm = platform_manager
        self.physics = physics_learner
        
        # 1. 맵 데이터
        self.waypoints = [] 
        self.visited_status = {} # {pid: last_visit_time}
        
        # 2. 전술 데이터 (사냥 효율)
        # 구조: {pid: {'kills': 0, 'time': 0, 'enter_time': 0}}
        self.sector_stats = defaultdict(lambda: {'kills': 0, 'time': 0, 'enter_time': 0})
        self.current_sector = -1
        self.best_sector = -1
        self.is_camping = False
        
        # 캠핑 기준: 10초 동안 5마리 이상 (초당 0.5마리) 잡으면 명당으로 인정
        self.CAMPING_THRESHOLD_KPS = 0.5 
        
        # 3. 경로 관리
        self.current_path = deque()
        self.target_node = None

    def build_graph(self):
        """맵 로드 시 실행: 발판 정보를 바탕으로 순찰 지점(Waypoint) 생성"""
        self.waypoints = []
        self.sector_stats.clear()
        self.visited_status.clear()
        self.is_camping = False
        self.best_sector = -1
        self.current_path.clear()
        
        if not self.pm or not self.pm.platforms: 
            return
        
        print(f"🗺️ [Navigator] 전술 지도 생성 중... ({len(self.pm.platforms)}개 구역)")
        
        for p in self.pm.platforms:
            pid = p.get('id', 0)
            y = p['y']
            margin = 30 # 발판 끝에서 안쪽으로 들어올 거리
            
            # 발판이 너무 짧으면 중앙 점 하나만 생성
            width = p['x_end'] - p['x_start']
            if width < 100:
                targets = [((p['x_start'] + p['x_end']) / 2, y)]
            else:
                # 발판을 3등분(좌, 중, 우)하여 이동 포인트로 잡음
                targets = [
                    (p['x_start'] + margin, y), 
                    ((p['x_start'] + p['x_end']) / 2, y),
                    (p['x_end'] - margin, y)
                ]
            
            for tx, ty in targets:
                # 좌표가 유효한지 재확인
                if p['x_start'] <= tx <= p['x_end']:
                    self.waypoints.append(Waypoint(tx, ty, pid))
            
            self.visited_status[pid] = 0

        print(f"✅ [Navigator] {len(self.waypoints)}개의 웨이포인트 생성 완료")

    def update_combat_stats(self, player_x, player_y, kill_increment):
        """
        [핵심] 몬스터 처치 시 호출됨.
        현재 구역의 사냥 효율(KPM)을 계산하고 '꿀자리'를 판별함.
        """
        plat = self.pm.get_current_platform(player_x, player_y)
        if not plat: return
        
        pid = plat['id']
        now = time.time()
        
        # 구역이 바뀌었으면 이전 구역 정산
        if self.current_sector != pid:
            if self.current_sector != -1:
                # 이전 구역 머문 시간 누적
                duration = now - self.sector_stats[self.current_sector]['enter_time']
                self.sector_stats[self.current_sector]['time'] += duration
                
            self.current_sector = pid
            self.sector_stats[pid]['enter_time'] = now
            
        # 킬 수 누적
        if kill_increment > 0:
            self.sector_stats[pid]['kills'] += kill_increment
            
            # 효율 계산 (Kills per Second)
            # 현재까지 누적된 시간 + 방금 들어와서 흐른 시간
            total_time = self.sector_stats[pid]['time'] + (now - self.sector_stats[pid]['enter_time'])
            
            # 데이터 신뢰성을 위해 최소 5초 이상 머문 곳만 평가
            if total_time > 5.0: 
                kps = self.sector_stats[pid]['kills'] / total_time
                
                # 명당 판단: 효율이 기준치를 넘고, 기존 최고 기록보다 좋다면 갱신
                # (기존 best가 있어도 더 좋은 곳이 나타나면 갈아탐)
                current_best_kps = 0
                if self.best_sector != -1:
                    ts = self.sector_stats[self.best_sector]
                    if ts['time'] > 0: current_best_kps = ts['kills'] / ts['time']

                if kps > self.CAMPING_THRESHOLD_KPS and kps > current_best_kps:
                    print(f"✨ [발견] 꿀자리를 찾았습니다! (ID: {pid}, 효율: {kps:.2f} kill/s)")
                    self.best_sector = pid

    def get_move_decision(self, player_x, player_y):
        """현재 상황에 맞는 이동 명령 반환 (이동 vs 캠핑)"""
        
        # 현재 위치 ID 확인
        curr_plat = self.pm.get_current_platform(player_x, player_y)
        curr_pid = curr_plat['id'] if curr_plat else -1

        # 1. 캠핑 모드 유지 확인
        if self.is_camping:
            # 명당 자리에 잘 있으면 -> 계속 캠핑
            if curr_pid == self.best_sector:
                return "None", "⛺ Camping" 
            else:
                # 밀려나거나 떨어졌으면 -> 다시 명당으로 복귀
                return self.navigate_to_pid(player_x, player_y, self.best_sector)

        # 2. 명당 자리를 알고 있다면? -> 그곳으로 이동
        if self.best_sector != -1:
            # 이미 명당에 도착했으면 캠핑 시작
            if curr_pid == self.best_sector:
                print(f"⛺ 명당(ID:{self.best_sector}) 도착! 제자리 사냥 시작.")
                self.is_camping = True
                self.current_path.clear()
                return "None", "Camping Start"
            
            # 명당으로 가는 길 안내
            print(f"🏃 꿀자리(ID:{self.best_sector})로 이동 중...")
            move, msg = self.navigate_to_pid(player_x, player_y, self.best_sector)
            return move, msg

        # 3. 정보가 부족하면 탐색(Patrol) 계속
        return self.patrol_mode(player_x, player_y)

    def navigate_to_pid(self, px, py, target_pid):
        """특정 발판(ID)으로 이동하는 경로 계산"""
        # 경로가 없거나, 경로의 목적지가 바뀌었으면 새로 계산
        if not self.current_path or (self.target_node and self.target_node.pid != target_pid):
            
            # 해당 ID를 가진 웨이포인트 중, 내 위치에서 가장 가까운 곳 선택
            candidates = [wp for wp in self.waypoints if wp.pid == target_pid]
            if not candidates: 
                return "None", "Invalid PID"
            
            target = min(candidates, key=lambda wp: math.hypot(wp.x - px, wp.y - py))
            
            print(f"🧭 경로 계산: ({int(px)},{int(py)}) -> ID:{target_pid}")
            path = self.find_path_astar(px, py, target.x, target.y)
            
            if path: 
                self.current_path = deque(path)
                self.target_node = target
            else:
                return "None", "Path Fail"
            
        if self.current_path:
            return self.current_path.popleft(), f"Nav({len(self.current_path)})"
            
        return "None", "Stuck"

    def patrol_mode(self, px, py):
        """정찰 모드: 안 가본 곳 위주로 돌아다님"""
        # 현재 위치 방문 기록 갱신
        plat = self.pm.get_current_platform(px, py)
        if plat: self.visited_status[plat['id']] = time.time()
        
        # 이동할 경로가 없으면 새로운 목표 선정
        if not self.current_path:
            target = self.get_next_patrol_target(px, py)
            if not target: return "None", "No Target"
            
            # print(f"🔍 정찰 목표 설정: ID {target.pid}")
            path = self.find_path_astar(px, py, target.x, target.y)
            if path: 
                self.current_path = deque(path)
                self.target_node = target
            else:
                # 못 가는 곳은 잠시 방문 처리해서 목표에서 제외
                self.visited_status[target.pid] = time.time()
            
        if self.current_path:
            return self.current_path.popleft(), "Patrol"
        return "None", "Idle"

    def get_next_patrol_target(self, player_x, player_y):
        """가장 오랫동안 방문하지 않은 곳 + 가까운 곳 점수 매겨서 선정"""
        now = time.time()
        best_target = None
        max_score = -float('inf')
        
        curr_plat = self.pm.get_current_platform(player_x, player_y)
        curr_pid = curr_plat['id'] if curr_plat else -1
        
        for wp in self.waypoints:
            # 1. 현재 있는 발판은 제외 (다른 곳으로 가야 함)
            if wp.pid == curr_pid: continue
            
            # 2. 점수 계산
            # Time Score: 오래 안 갈수록 점수 높음
            time_score = now - self.visited_status.get(wp.pid, 0)
            
            # Dist Score: 너무 멀면 감점 (가까운 곳부터 탐색)
            dist = math.hypot(wp.x - player_x, wp.y - player_y)
            dist_score = dist * 0.5 
            
            final_score = time_score - dist_score
            
            if final_score > max_score:
                max_score = final_score
                best_target = wp
                
        return best_target

    def find_path_astar(self, start_x, start_y, goal_x, goal_y):
        """물리 엔진 예측을 활용한 A* 알고리즘"""
        if not self.physics.model: return []

        open_list = []
        closed_set = set()
        
        # 시작 노드
        h_start = math.hypot(goal_x - start_x, goal_y - start_y)
        heapq.heappush(open_list, PathNode(start_x, start_y, 0, h_start))
        
        steps = 0
        max_steps = 300 # 연산량 제한
        
        best_node_so_far = None
        min_dist_to_goal = float('inf')
        
        while open_list and steps < max_steps:
            steps += 1
            curr = heapq.heappop(open_list)
            
            # 목표와의 거리 확인
            dist = math.hypot(goal_x - curr.x, goal_y - curr.y)
            
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                best_node_so_far = curr
            
            # 도착 판정 (30px 이내면 도착으로 간주)
            if dist < 30:
                return self.reconstruct_path(curr)
            
            # 방문 체크 (20px 그리드 단위)
            state_key = (int(curr.x // 20), int(curr.y // 20))
            if state_key in closed_set: continue
            closed_set.add(state_key)
            
            # 물리 엔진을 통한 다음 위치 예측
            is_grounded = (self.pm.get_current_platform(curr.x, curr.y) is not None)
            
            for action in self.physics.possible_actions:
                dx, dy = self.physics.get_displacement(action, is_grounded)
                
                # [★핵심 수정 1] 강제 중력 부여 (Gravity Injection)
                # 공중에 떠 있다면(not is_grounded), 강제로 아래쪽(y+) 힘을 가함
                if not is_grounded:
                    dy += 8.0 # 중력 가속도 시뮬레이션 (값이 클수록 뚝 떨어짐)
                
                # [★핵심 수정 2] 수평 과속 방지 (핵 이동 방지)
                # 만약 물리 엔진이 비정상적으로 빠른 X축 이동을 예측하면 패널티 부여
                if abs(dx) > 25: # 플래시 점프 등으로 너무 빠르면
                     dx *= 0.8   # 속도를 깎아서 보수적으로 판단

                if abs(dx)<2 and abs(dy)<2: continue
                
                nx, ny = curr.x+dx, curr.y+dy
                if not (0<=nx<=1366 and -200<=ny<=1000): continue
                
                # 비용 계산 (포물선을 그리면 거리가 늘어나므로 자연스레 비용 증가)
                cost = math.hypot(dx, dy)
                if dy < 0: cost *= 1.5 # 위로 가는 동작은 비용을 더 줘서 남발 방지
                
                ng = curr.g + cost
                if ng + math.hypot(goal_x-nx, goal_y-ny) > curr.h + 500: continue
                heapq.heappush(open_list, PathNode(nx, ny, ng, math.hypot(goal_x-nx, goal_y-ny), curr, action))
                
        # 경로를 못 찾았으면, 그나마 가장 가까이 간 경로라도 반환
        if best_node_so_far and min_dist_to_goal < 200:
            return self.reconstruct_path(best_node_so_far)
            
        return [] # 실패

    def reconstruct_path(self, node):
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))