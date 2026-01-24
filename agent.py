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
        self.fixed_pattern = [("left+jump+q", 0.6), ("right+jump+q", 0.6), ("q", 0.5)]
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
        self.job_encoder = None
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
        self.last_kill_count = 0; self.prev_px = -1; self.prev_py = -1
        self.last_pos = (0, 0); self.last_pos_time = time.time(); self.stuck_count = 0
        self.current_unstuck_action = None; self.unstuck_timer = 0
        self.my_job_id = 0
        
        # 행동 제어 타이머
        self.busy_until = 0 
        self.hold_key_until = 0

        # [신규] 착지 대기 로직
        self.waiting_for_landing = False
        self.jump_start_time = 0
        self.landing_stable_frames = 0  

        # 행동별 쿨타임 (애니메이션 시간 고려)
        self.action_cooldowns = {
            'up+jump': 0.1,      # 쿨타임은 짧게 잡고, 착지 대기 로직으로 제어
            'down+jump': 0.1,    
            'double_jump': 0.1,  
            'jump': 0.1,         
            'rope': 1.2,         
            'sub_attack': 0.8,   
            'ultimate': 1.5      
        }

        self.action_lock_until = 0 
        self.last_action_name = "None"

    def load_lstm(self, file_path):
        try:
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            self.scaler = checkpoint['scaler']
            self.encoder = checkpoint['encoder']
            self.job_encoder = checkpoint.get('job_encoder', None)
            if self.job_encoder:
                try: self.my_job_id = self.job_encoder.transform(['Kinesis'])[0]
                except: self.my_job_id = 0
                print(f"🆔 Job Detected: Kinesis (ID: {self.my_job_id})")
            self.feature_cols = checkpoint.get('feature_cols', self.feature_cols)
            self.seq_length = checkpoint.get('seq_length', 10)
            input_size = checkpoint.get('input_size', len(self.feature_cols))
            num_jobs_found = len(self.job_encoder.classes_) if self.job_encoder else 1
            self.lstm_model = LSTMModel(
                input_size=input_size, hidden_size=256, num_layers=3,
                num_classes=len(self.encoder.classes_), num_jobs=num_jobs_found, dropout=0.3
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
        self.busy_until = 0; self.hold_key_until = 0
        self.waiting_for_landing = False

    def on_map_change(self, map_json_path):
        self.pm.load_platforms(map_json_path); self.navigator.build_graph(map_json_path)
        print(f"🗺️ Agent: 맵 정보 갱신 완료")

    def check_is_stuck(self, px, py):
        # [수정] 설치기 사용 중이거나 쿨타임 대기 중일 때는 고착 감지 건너뛰기
        # hold_key_until: 키를 누르고 있는 시간 (이때는 당연히 안 움직임)
        # busy_until: 후딜레이 시간 (이때도 안 움직일 수 있음)
        if time.time() < self.hold_key_until or time.time() < self.busy_until:
            self.last_pos_time = time.time() # 시간을 갱신해서 카운트 리셋
            self.stuck_count = 0             # 카운트도 0으로 초기화 (중요)
            return False

        now = time.time()
        if now - self.last_pos_time > 2.0:
            dist = abs(px - self.last_pos[0]) + abs(py - self.last_pos[1])
            self.last_pos = (px, py); self.last_pos_time = now
            if dist < 20: self.stuck_count += 1
            else: self.stuck_count = 0
        
        return self.stuck_count >= 3

    def apply_cooldown(self, action_name):
        """행동에 맞는 쿨타임 적용"""
        base_cd = 0
        for key, cd in self.action_cooldowns.items():
            if key in action_name:
                base_cd = max(base_cd, cd)
        
        if base_cd > 0:
            final_cd = base_cd + random.uniform(-0.05, 0.05)
            self.busy_until = time.time() + final_cd

    def get_action(self, px, py, entropy, pid, ult_ready, sub_ready, dist_left=0, dist_right=0, current_kill_count=0, job_id=None, vision=None, frame=None):
        
        # 0. Job ID Update
        if job_id is not None: 
            self.my_job_id = job_id

        # ---------------------------------------------------------
        # [1. Information Update] (Run First)
        # ---------------------------------------------------------
        
        # (1) Calculate Movement (dx, dy)
        if self.prev_px == -1: 
            dx, dy = 0, 0
        else: 
            dx = px - self.prev_px
            dy = py - self.prev_py
        
        self.prev_px = px
        self.prev_py = py

        # (2) Update Kill Count & Combat Stats
        kill_diff = max(0, current_kill_count - self.last_kill_count)
        self.last_kill_count = current_kill_count
        
        if kill_diff > 0: 
            self.navigator.update_combat_stats(px, py, kill_diff)
            self.gen_manager.update_kill()

        # ---------------------------------------------------------
        # [2. Status Control & Locking Logic]
        # ---------------------------------------------------------

        now = time.time()

        # A. Holding Key (e.g., Installing Skill)
        if now < self.hold_key_until:
            return "sub_attack", "⚡ Holding Skill..."

        # B. Global Cooldown (Busy)
        if now < self.busy_until:
            return "None", "⏳ Acting..."

        # C. Action Locking (Anti-Jitter)
        # Prevents changing decisions too quickly (e.g., moving left -> stopping -> moving left)
        # Exceptions: Wall collision or landing on a new platform overrides the lock.
        if now < self.action_lock_until:
            # If we are stuck or hit a wall, break the lock
            if dx == 0 and "jump" not in self.last_action_name:
                pass # Let new logic run
            else:
                return self.last_action_name, "🔒 Locking..."

        # D. Landing Wait Logic
        if self.waiting_for_landing:
            # Timeout
            if now - self.jump_start_time > 1.5:
                self.waiting_for_landing = False
                self.landing_stable_frames = 0
            
            # Platform Detected
            elif pid != -1:
                self.waiting_for_landing = False
                self.landing_stable_frames = 0
                self.busy_until = now + 0.1
            
            # Physics Stop (Y-axis stable)
            elif abs(dy) <= 1 and abs(dx) <= 2: 
                 self.landing_stable_frames += 1
                 if self.landing_stable_frames >= 5:
                     self.waiting_for_landing = False
                     self.landing_stable_frames = 0
            else:
                 self.landing_stable_frames = 0
                 return "None", "🦅 Gliding..."

        # E. Unstuck Logic
        if self.check_is_stuck(px, py):
            self.action_queue.clear()
            self.gen_manager.pattern_queue.clear()
            self.gen_manager.current_pattern_action = None
            
            if self.current_unstuck_action and (now - self.unstuck_timer < 1.5):
                return self.current_unstuck_action, f"🚧 Unstuck! (Trying...)"
            
            floor_y = getattr(self.navigator.patrol, 'map_floor_y', 999)
            opts = ["left+jump", "right+jump", "up+jump", "down+jump"]
            
            # Context-aware Unstuck
            if py > floor_y - 15: opts = ["left+jump", "right+jump", "up+jump"]
            if px < 50: opts = [o for o in opts if "right" in o or "up" in o]
            elif px > 1300: opts = [o for o in opts if "left" in o or "up" in o]
            
            if not opts: opts = ["up+jump"]
            
            self.current_unstuck_action = random.choice(opts)
            self.unstuck_timer = now
            self.busy_until = now + 0.5 
            
            if "jump" in self.current_unstuck_action:
                self.waiting_for_landing = True
                self.jump_start_time = now
                
            return self.current_unstuck_action, f"🚧 Unstuck! (New)"
        else: 
            self.current_unstuck_action = None

        # ---------------------------------------------------------
        # [3. Decision Making]
        # ---------------------------------------------------------

        # A. Gen Cycle Patterns
        cycle_state = self.gen_manager.check_cycle()
        if cycle_state == "PRE_GEN":
            if not self.gen_manager.pattern_queue and not self.gen_manager.current_pattern_action:
                self.gen_manager.start_pattern()
            
            if self.gen_manager.current_pattern_action:
                if now - self.gen_manager.pattern_timer < self.gen_manager.current_pattern_duration:
                    return self.gen_manager.current_pattern_action, "🔄 Pattern Exec"
                else: self.gen_manager.current_pattern_action = None
            
            if self.gen_manager.pattern_queue:
                act, dur = self.gen_manager.pattern_queue.popleft()
                self.gen_manager.current_pattern_action = act
                self.gen_manager.current_pattern_duration = dur
                self.gen_manager.pattern_timer = now
                self.busy_until = now + 0.1 
                return act, "🔄 Pattern Start"

        if self.action_queue: 
            act = self.action_queue.popleft()
            self.apply_cooldown(act) 
            return act, f"Seq"

        # B. LSTM Model Inference
        lstm_attack_cmd = "None"; lstm_raw_cmd = "None"
        if self.lstm_model:
            try:
                input_data = {'player_x': px, 'player_y': py, 'delta_x': dx, 'delta_y': dy, 'entropy': entropy, 'platform_id': pid, 'ult_ready': ult_ready, 'sub_ready': sub_ready, 'inv_dist_left': dist_left, 'inv_dist_right': dist_right}
                df = pd.DataFrame([input_data])
                for col in self.feature_cols: 
                    if col not in df.columns: df[col] = 0
                feats_scaled = self.scaler.transform(df[self.feature_cols])
                self.history.append(feats_scaled[0])
                if len(self.history) == self.seq_length:
                    inp = torch.FloatTensor(np.array([self.history])).to(self.device)
                    job_tensor = torch.LongTensor([self.my_job_id]).to(self.device)
                    with torch.no_grad():
                        out = self.lstm_model(inp, job_tensor)
                        if out.dim() == 3: out = out[:, -1, :]
                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                        # Top 3 Actions
                        top_idxs = np.argsort(probs)[::-1][:3] 
                        actions = self.encoder.inverse_transform(top_idxs)
                        for act in actions:
                            act = act.lower()
                            if act not in ['left', 'right', 'up', 'down', 'none', 'idle', 'jump', 'double_jump']:
                                lstm_attack_cmd = act; break
                        lstm_raw_cmd = self.encoder.inverse_transform([np.argmax(probs)])[0]
            except Exception: pass

        # C. Tactical Navigator
        nav_action, nav_msg = self.navigator.get_move_decision(px, py, install_ready=(sub_ready==1))
        
        if cycle_state == "SEARCH" and (nav_action == "None" or "Camping" in nav_msg):
             nav_action = self.navigator.patrol_mode(px, py)
             nav_msg = "Searching..."

        # ---------------------------------------------------------
        # [4. Action Merging & Final Selection]
        # ---------------------------------------------------------
        final_action = []
        final_msg = nav_msg
        
        # [Install Logic]
        if "Positioned for Install" in nav_msg and sub_ready == 1:
            skill_name = "fountain"
            if self.navigator.patrol.next_skill_to_use:
                skill_name = self.navigator.patrol.next_skill_to_use.name

            # Check Vision for Cooldown (Success)
            if vision and frame is not None:
                if vision.is_skill_on_cooldown(skill_name, frame):
                    self.navigator.notify_install_success()
                    return "None", "✅ Already Installed (Skip)"

            # Execute Install
            base_dur = 0.55; sigma = 0.03
            press_duration = base_dur + random.gauss(0, sigma)
            
            self.hold_key_until = now + press_duration
            self.busy_until = now + 1.0 # Wait longer for animation
            return "sub_attack", f"Deploy (Hold {press_duration:.2f}s)"

        # 1. Movement Selection
        if nav_action != "None":
            final_action.append(nav_action)
            # Double Jump Logic
            if "jump" not in nav_action and random.random() < 0.6: 
                final_action.append("double_jump")
        elif lstm_raw_cmd in ['left', 'right', 'up', 'down']:
            final_action.append(lstm_raw_cmd)
        
        # 2. Attack Selection
        attack_candidate = None
        if lstm_attack_cmd != "None":
            attack_candidate = lstm_attack_cmd.replace('left', '').replace('right', '').replace('up', '').replace('down', '').replace('+', '')
        elif random.random() < 0.6: 
            attack_candidate = 'r'

        if attack_candidate:
            final_action.append(attack_candidate)
            final_msg += f" + ATK({attack_candidate})"

        if not final_action: return "None", "Idle"
        
        result_action = "+".join(final_action)
        
        # ---------------------------------------------------------
        # [5. Apply Action Lock]
        # ---------------------------------------------------------
        # If the action changed, lock it for a short duration to prevent jitter
        if result_action != self.last_action_name:
            if "left" in result_action or "right" in result_action:
                self.action_lock_until = now + 0.12 # Move lock
            elif "jump" in result_action:
                self.action_lock_until = now + 0.08 # Jump lock
            
            self.last_action_name = result_action

        # Cooldowns & Landing Logic
        self.apply_cooldown(result_action)
        
        if "jump" in result_action and "up" in result_action:
             self.waiting_for_landing = True
             self.landing_stable_frames = 0
             self.jump_start_time = now
             final_msg += " (Wait Land)"
        
        if "double_jump" in result_action or ("jump" in result_action and "down" in result_action):
             self.waiting_for_landing = True
             self.landing_stable_frames = 0
             self.jump_start_time = now
             final_msg += " (Wait Land)"

        return result_action, final_msg