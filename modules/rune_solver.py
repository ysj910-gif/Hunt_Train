import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import math
import time
import heapq
from collections import deque

try:
    from platform_manager import PlatformManager
except ImportError:
    PlatformManager = None

# [1] 물리 엔진 모델
class HybridPhysicsNet(nn.Module):
    def __init__(self, num_actions):
        super(HybridPhysicsNet, self).__init__()
        self.physics_params = nn.Embedding(num_actions, 3)
        self.physics_params.weight.data.uniform_(0.1, 1.0)
        self.action_emb = nn.Embedding(num_actions, 8)
        self.residual_net = nn.Sequential(
            nn.Linear(8 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, action_idx, is_grounded):
        if is_grounded.dim() > 1:
            is_grounded = is_grounded.squeeze(1)

        params = self.physics_params(action_idx)
        phys_vx = params[:, 0] * 10.0
        phys_vy = params[:, 1] * 10.0
        gravity = params[:, 2] * 5.0 * (1.0 - is_grounded)
        
        base_dx = phys_vx
        base_dy = phys_vy + gravity
        base_move = torch.stack([base_dx, base_dy], dim=1)
        
        emb = self.action_emb(action_idx)
        cat_ground = is_grounded.unsqueeze(1)
        cat = torch.cat([emb, cat_ground], dim=1)
        residual = self.residual_net(cat)
        
        return base_move + residual

# [2] 물리 학습기
class PhysicsLearner:
    def __init__(self):
        self.model = None
        self.encoder = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.possible_actions = []

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.encoder = checkpoint['encoder']
            self.possible_actions = list(self.encoder.classes_)
            num_actions = len(self.possible_actions)
            
            self.model = HybridPhysicsNet(num_actions).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            print(f"✅ [PhysicsLearner] 모델 로드 완료 (행동 {num_actions}개)")
            return True
        except Exception as e:
            print(f"❌ [PhysicsLearner] 로드 에러: {e}")
            return False

    def get_displacement(self, action_name, is_grounded=True):
        if not self.model or not self.encoder: return (0, 0)
        try:
            act_idx = self.encoder.transform([action_name])[0]
        except: return (0, 0)
        
        act_tensor = torch.LongTensor([act_idx]).to(self.device)
        state_tensor = torch.tensor([1.0 if is_grounded else 0.0], device=self.device)
        
        with torch.no_grad():
            pred = self.model(act_tensor, state_tensor)
            
        dx, dy = pred[0].cpu().numpy()
        return float(dx), float(dy)

# [3] 경로 탐색 (A*)
class PathNode:
    def __init__(self, x, y, g, h, parent=None, action=None):
        self.x = x; self.y = y; self.g = g; self.h = h
        self.f = g + h; self.parent = parent; self.action = action
    def __lt__(self, other): return self.f < other.f

class PathFinder:
    def __init__(self, learner, platform_mgr):
        self.learner = learner
        self.pm = platform_mgr
        
    def find_path(self, start_x, start_y, goal_x, goal_y):
        if not self.learner.model:
            return None 

        open_list = []
        closed_set = set()
        
        h_start = math.hypot(goal_x - start_x, goal_y - start_y)
        heapq.heappush(open_list, PathNode(start_x, start_y, 0, h_start))
        
        best_node = None
        min_dist_to_goal = float('inf')
        
        steps = 0
        max_steps = 300 
        
        while open_list and steps < max_steps:
            steps += 1
            current = heapq.heappop(open_list)
            
            dist = math.hypot(goal_x - current.x, goal_y - current.y)
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                best_node = current
            
            if dist < 15: 
                return self.reconstruct_path(current)
            
            state_key = (int(current.x // 20), int(current.y // 20))
            if state_key in closed_set: continue
            closed_set.add(state_key)
            
            is_grounded = True
            if self.pm: is_grounded = self.pm.get_current_platform(current.x, current.y) is not None
            
            for action in self.learner.possible_actions:
                dx, dy = self.learner.get_displacement(action, is_grounded)
                
                if abs(dx) < 2 and abs(dy) < 2: continue
                
                nx, ny = current.x + dx, current.y + dy
                if not (0 <= nx <= 1366): continue
                
                cost = math.hypot(dx, dy)
                if abs(dy) > 5: cost *= 1.5 
                
                ng = current.g + cost
                nh = math.hypot(goal_x - nx, goal_y - ny)
                
                if nh > h_start + 200: continue
                
                heapq.heappush(open_list, PathNode(nx, ny, ng, nh, current, action))
                
        return self.reconstruct_path(best_node) if best_node else None

    def reconstruct_path(self, node):
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))

# [4] 룬 매니저 (핵심 수정됨)
class RuneManager:
    def __init__(self):
        self.learner = PhysicsLearner()
        self.platform_mgr = PlatformManager() if PlatformManager else None
        self.path_finder = PathFinder(self.learner, self.platform_mgr)
        
        self.lower_purple = np.array([130, 80, 80])
        self.upper_purple = np.array([150, 255, 255])
        self.rune_pos = None 
        self.last_scan_time = 0
        
        # [설정] 기본 탐색 간격 (60초)
        self.scan_interval = 60.0 
        
        self.path_queue = deque()
        self.last_path_calc_time = 0
        self.recalc_interval = 2.0 

    def load_physics(self, filepath):
        return self.learner.load_model(filepath)
    
    def load_map(self, map_file):
        if self.platform_mgr: self.platform_mgr.load_platforms(map_file)

    def scan_for_rune(self, minimap_img):
        """
        룬 탐색 로직 (스마트 주기 적용)
        - 룬이 없을 때: 60초마다 탐색
        - 룬을 추적 중일 때: 3초마다 재확인 (사라짐 감지)
        """
        now = time.time()
        
        # [핵심] 현재 상태에 따라 검사 주기 변경
        current_interval = self.scan_interval # 기본 60초
        if self.rune_pos is not None:
            current_interval = 3.0 # 추적 중엔 3초마다 검사
            
        # 쿨타임 체크
        if now - self.last_scan_time < current_interval: 
            return self.rune_pos
        
        self.last_scan_time = now
        if minimap_img is None: return None

        # 이미지 처리
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_purple, self.upper_purple)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_pos = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 500:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    best_pos = (cx, cy)
                    break
        
        # [상태 업데이트 및 로깅]
        prev_pos = self.rune_pos
        self.rune_pos = best_pos
        
        if best_pos:
            # 새로 발견했을 때만 로그 출력
            if prev_pos is None:
                print(f"[{time.strftime('%H:%M:%S')}] ✨ 룬 발견! 위치: {best_pos}")
        else:
            # 룬이 있었는데 사라졌을 때 로그 출력
            if prev_pos is not None:
                print(f"[{time.strftime('%H:%M:%S')}] 🗑️ 룬이 사라졌습니다. (추적 중지)")
                self.path_queue.clear() # 이동 경로 취소
                
        return best_pos

    def get_move_action(self, player_x, player_y):
        """이동 로직"""
        if not self.rune_pos: return None, "No Rune"

        rx, ry = self.rune_pos
        dx = rx - player_x
        dy = ry - player_y
        dist = math.hypot(dx, dy)
        
        # 1. 도착 판정
        if dist < 5.0:
            print("✨ 룬 도착! 상호작용 시도")
            self.rune_pos = None 
            self.path_queue.clear()
            return "interact", "🎯 Arrived"
        
        # 2. 미세 조정
        if dist < 20.0:
            self.path_queue.clear()
            if abs(dx) > 3.0: return ("right" if dx > 0 else "left"), "Micro-X"
            if abs(dy) > 3.0: return ("down" if dy > 0 else "up"), "Micro-Y"

        # 3. A* 경로 탐색
        now = time.time()
        if (not self.path_queue) and (now - self.last_path_calc_time > self.recalc_interval):
            # print(f"🧩 A* 경로 계산 시도... (거리: {int(dist)})")
            path = self.path_finder.find_path(player_x, player_y, rx, ry)
            
            if path:
                self.path_queue = deque(path)
                self.last_path_calc_time = now
            else:
                # print("⚠️ 경로 탐색 실패 -> Fallback 모드")
                self.path_queue.clear()

        # 4. 큐 실행
        if self.path_queue:
            return self.path_queue.popleft(), f"A*({len(self.path_queue)})"

        # 5. [Fallback] 단순 이동
        if dy < -30: 
            if abs(dx) > 20: return ("right+jump" if dx > 0 else "left+jump"), "Fallback-Jump"
            return "up+jump", "Fallback-UpJump"
            
        elif dy > 50:
            return "down+jump", "Fallback-Down"
            
        if abs(dx) > 5:
            return ("right" if dx > 0 else "left"), "Fallback-Run"
            
        return None, "Stuck?"