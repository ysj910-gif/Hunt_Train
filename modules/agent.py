import torch
import numpy as np
import pandas as pd
import time
import random
from collections import deque
from modules.model import LSTMModel
from modules.rune_solver import PhysicsLearner
from modules.navigator import TacticalNavigator
from platform_manager import PlatformManager

class GenCycleManager:
    def __init__(self):
        self.GEN_INTERVAL = 7.5
        self.last_kill_time = time.time()
        self.has_performed_pattern = False 
        self.fixed_pattern = [
            ("left+jump+q", 0.6), ("right+jump+q", 0.6), ("q", 0.5)
        ]
        self.pattern_queue = deque()
        self.current_pattern_action = None
        self.current_pattern_duration = 0
        self.pattern_timer = 0

    def update_kill(self):
        self.last_kill_time = time.time()
        self.has_performed_pattern = False 

    def check_cycle(self):
        elapsed = time.time() - self.last_kill_time
        if elapsed < 2.0: return "COMBAT"
        elif elapsed < 6.5: return "WAITING"
        else: return "SEARCH" if self.has_performed_pattern else "PRE_GEN"

    def start_pattern(self):
        if not self.pattern_queue:
            for act, dur in self.fixed_pattern:
                self.pattern_queue.append((act, dur))
            self.has_performed_pattern = True

class BotAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model = None; self.scaler = None; self.encoder = None
        self.action_queue = deque()
        self.seq_length = 10 
        self.history = deque(maxlen=self.seq_length)
        self.feature_cols = [
            'player_x', 'player_y', 'delta_x', 'delta_y', 
            'entropy', 'platform_id', 'ult_ready', 'sub_ready', 
            'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right', 
            'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
        ]
        self.pm = PlatformManager()
        self.physics = PhysicsLearner()
        try: self.physics.load_model("physics_hybrid_model.pth")
        except: pass
        self.navigator = TacticalNavigator(self.pm, self.physics)
        self.gen_manager = GenCycleManager()
        self.mode = "HYBRID" 
        self.last_kill_count = 0; self.prev_px = 0; self.prev_py = 0
        self.last_pos = (0, 0); self.last_pos_time = time.time(); self.stuck_count = 0
        self.current_unstuck_action = None; self.unstuck_timer = 0

    def load_lstm(self, file_path):
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.scaler = checkpoint['scaler']; self.encoder = checkpoint['encoder']
            self.feature_cols = checkpoint.get('feature_cols', self.feature_cols)
            self.seq_length = checkpoint.get('seq_length', 10)
            self.lstm_model = LSTMModel(
                checkpoint.get('input_size', len(self.feature_cols)),
                checkpoint.get('hidden_size', 256),
                checkpoint.get('num_layers', 3),
                checkpoint.get('num_classes', 10),
                checkpoint.get('future_steps', 5),
                checkpoint.get('dropout', 0.3)
            ).to(self.device)
            self.lstm_model.load_state_dict(checkpoint['model_state'])
            self.lstm_model.eval()
            self.history = deque(maxlen=self.seq_length)
            return True, "LSTM Loaded"
        except Exception as e: return False, str(e)
    def load_rf(self, f): return False, "RF Disabled"
    def reset_history(self):
        self.history.clear(); self.action_queue.clear()
        self.last_kill_count = 0; self.gen_manager.update_kill()
        self.current_unstuck_action = None
    def on_map_change(self, map_json_path):
        self.pm.load_platforms(map_json_path); self.navigator.build_graph()
        print(f"🗺️ Agent: 맵 정보 갱신 완료")

    def check_is_stuck(self, px, py):
        now = time.time()
        if now - self.last_pos_time > 1.5:
            dist = abs(px - self.last_pos[0]) + abs(py - self.last_pos[1])
            self.last_pos = (px, py); self.last_pos_time = now
            if dist < 20 and px > 0: self.stuck_count += 1
            else: self.stuck_count = 0
        return self.stuck_count >= 2

    def get_action(self, px, py, entropy, pid, ult_ready, sub_ready, dist_left=0, dist_right=0, current_kill_count=0):
        # 1. 고착 탈출 (Persistent)
        if self.check_is_stuck(px, py):
            self.action_queue.clear(); self.gen_manager.pattern_queue.clear()
            self.gen_manager.current_pattern_action = None
            now = time.time()
            if self.current_unstuck_action and (now - self.unstuck_timer < 1.0):
                return self.current_unstuck_action, f"🚧 Unstuck! (Trying...)"
            
            opts = ["left+jump", "right+jump", "up+jump", "down+jump"]
            if px < 50: opts = ["right+jump", "up+jump"]
            elif px > 1300: opts = ["left+jump", "up+jump"]
            self.current_unstuck_action = random.choice(opts)
            self.unstuck_timer = now
            return self.current_unstuck_action, f"🚧 Unstuck! (New)"
        else: self.current_unstuck_action = None

        # 2. 정보 업데이트
        kill_diff = max(0, current_kill_count - self.last_kill_count)
        self.last_kill_count = current_kill_count
        if kill_diff > 0: self.navigator.update_combat_stats(px, py, kill_diff); self.gen_manager.update_kill() 
        if px == 0 or self.prev_px == 0: dx, dy = 0, 0
        else: dx = px - self.prev_px; dy = py - self.prev_py
        self.prev_px = px; self.prev_py = py

        # 3. 젠 사이클
        cycle_state = self.gen_manager.check_cycle()
        if cycle_state == "PRE_GEN":
            if not self.gen_manager.pattern_queue and not self.gen_manager.current_pattern_action:
                self.gen_manager.start_pattern()
            now = time.time()
            if self.gen_manager.current_pattern_action:
                if now - self.gen_manager.pattern_timer < self.gen_manager.current_pattern_duration:
                    return self.gen_manager.current_pattern_action, "🔄 Pattern Exec"
                else: self.gen_manager.current_pattern_action = None
            if self.gen_manager.pattern_queue:
                act, dur = self.gen_manager.pattern_queue.popleft()
                self.gen_manager.current_pattern_action = act
                self.gen_manager.current_pattern_duration = dur
                self.gen_manager.pattern_timer = now
                return act, "🔄 Pattern Start"
        elif cycle_state == "WAITING":
            nav_action, nav_msg = self.navigator.get_move_decision(px, py)
            if "Camping" in nav_msg: return "None", "Waiting Gen..."
            if nav_action != "None": return nav_action, "Go to Position"

        # 4. 전투 모드 (스킬 주입 로직 추가)
        if self.action_queue: return self.action_queue.popleft(), f"Seq"

        # LSTM
        lstm_skill_intent = "None"
        if self.lstm_model:
            try:
                # (입력 데이터 구성 생략 - 이전과 동일)
                input_data = {'player_x': px, 'player_y': py, 'delta_x': dx, 'delta_y': dy, 'entropy': entropy, 'platform_id': pid, 'ult_ready': ult_ready, 'sub_ready': sub_ready}
                df = pd.DataFrame([input_data])
                for col in self.feature_cols: 
                    if col not in df.columns: df[col] = 0
                feats_scaled = self.scaler.transform(df[self.feature_cols]); self.history.append(feats_scaled[0])
                if len(self.history) == self.seq_length:
                    inp = torch.FloatTensor(np.array([self.history])).to(self.device)
                    with torch.no_grad():
                        out = self.lstm_model(inp); _, preds = torch.max(out, 2)
                        actions = self.encoder.inverse_transform(preds.squeeze(0).cpu().numpy())
                        for a in actions:
                            a = a.lower()
                            if a not in ['left', 'right', 'up', 'down', 'jump', 'space', 'none']: lstm_skill_intent = a; break
            except: pass

        nav_action, nav_msg = self.navigator.get_move_decision(px, py)
        if cycle_state == "SEARCH" and "Camping" in nav_msg:
             nav_action = self.navigator.patrol_mode(px, py); nav_msg = "Searching..."

        if "Camping" in nav_msg:
            if lstm_skill_intent != "None": return lstm_skill_intent, "Camp(LSTM)"
            if sub_ready == 1: return "q", "Camp+Auto"
            return "None", "Camping"

        if nav_action != "None":
            combo = [nav_action]
            if random.random() < 0.8: combo.append("jump")
            
            # [핵심] 이동 중 공격 강제 주입 ('r' 키가 일반기라고 가정)
            # LSTM이 특별한 스킬을 쓰라고 한 게 아니면, 기본적으로 'r'을 섞음
            if lstm_skill_intent != "None":
                skill_key = lstm_skill_intent.replace("left", "").replace("right", "").replace("up", "").replace("down", "").replace("+", "")
                if skill_key: combo.append(skill_key)
            else:
                # 80% 확률로 공격 섞기 (너무 100%면 이동 속도 느려질 수 있음)
                if random.random() < 0.8: 
                    combo.append("r") 
                    nav_msg += "+Atk"
            
            return "+".join(combo), nav_msg
            
        return "None", "Idle"