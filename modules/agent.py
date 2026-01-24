import torch
import numpy as np
import pandas as pd
import time
import random
import json
import warnings
from collections import deque
from modules.model import LSTMModel
from modules.rune_solver import PhysicsLearner
from modules.navigator import TacticalNavigator
from platform_manager import PlatformManager

class GenCycleManager:
    def __init__(self, skill_key_map=None):
        self.GEN_INTERVAL = 7.5
        self.last_kill_time = time.time()
        self.has_performed_pattern = False 
        
        # [수정] 설치기 키 맵 저장 및 동적 패턴 생성
        self.skill_key_map = skill_key_map or {}
        self.build_gen_pattern()
        
        self.pattern_queue = deque()
        self.current_pattern_action = None
        self.current_pattern_duration = 0
        self.pattern_timer = 0
    
    def build_gen_pattern(self):
        """설치기 키 맵을 기반으로 젠 패턴 동적 생성"""
        self.fixed_pattern = []
        
        # 모든 설치기에 대해 좌/우 점프 + 설치 패턴 생성
        for skill_name, key in self.skill_key_map.items():
            self.fixed_pattern.append((f"left+jump+{key}", 0.6))
            self.fixed_pattern.append((f"right+jump+{key}", 0.6))
            self.fixed_pattern.append((key, 0.5))
        
        # 설치기가 없으면 기본 패턴 (이동만)
        if not self.fixed_pattern:
            self.fixed_pattern = [("left+jump", 0.4), ("right+jump", 0.4)]

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
        
        self.mode = "HYBRID" 

        # [전면 수정] Config 파일로부터 모든 스킬/키 정보 로드
        self.skill_key_map = {}       # 모든 스킬명 -> 키 매핑 (예: {"싸이킥 불릿": "r", "fountain": "4"})
        self.install_key_map = {}     # 설치기 전용 (예: {"fountain": "4", "janus": "5"})
        self.attack_key = "r"         # 기본 공격 키
        self.jump_key = "e"           # 점프 키
        self.skill_cooldowns = {}     # 스킬별 쿨타임 (예: {"체크메이트": 30.0})
        
        self.load_keys_from_config()  # Config 로드
        self.gen_manager = GenCycleManager(self.install_key_map)  # 설치기 맵 전달

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

        # 행동별 쿨타임 (애니메이션 시간 고려) - Config에서 로드한 값으로 덮어쓸 수 있음
        self.action_cooldowns = {
            'up+jump': 0.1,
            'down+jump': 0.1,    
            'double_jump': 0.1,  
            'jump': 0.1,         
            'rope': 1.2,         
            'sub_attack': 0.8,   
            'ultimate': 1.5      
        }

        self.action_lock_until = 0 
        self.last_action_name = "None"

    def load_keys_from_config(self):
        """hunter_config.json에서 스킬/키 정보 전면 로드"""
        try:
            with open("hunter_config.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print(f"[DEBUG] Config 파일 로드 성공")
            
            # 1. 현재 직업 확인
            job_name = data.get("last_job", "Kinesis")
            print(f"[DEBUG] 직업: {job_name}")
            
            job_settings = data.get("job_settings", {}).get(job_name, {})
            
            if not job_settings:
                print(f"⚠️ [Agent] job_settings에 '{job_name}' 정보가 없습니다!")
                # 폴백: mapping에서 로드 시도
                mapping = data.get("mapping", {})
                for skill_name, skill_info in mapping.items():
                    key = skill_info.get("key")
                    if key:
                        self.skill_key_map[skill_name] = key
                        if "불릿" in skill_name:
                            self.attack_key = key
                        elif "jump" in skill_name.lower():
                            self.jump_key = key
                print(f"[DEBUG] mapping에서 {len(self.skill_key_map)}개 스킬 로드")
                return
            
            # 2. 스킬 정보 로드 (skills)
            skills = job_settings.get("skills", {})
            print(f"[DEBUG] skills 섹션: {len(skills)}개")
            
            for skill_name, skill_info in skills.items():
                key = skill_info.get("key")
                cd = skill_info.get("cd", 0.0)
                
                if key:
                    self.skill_key_map[skill_name] = key
                    
                    # 쿨타임 저장
                    if cd > 0:
                        self.skill_cooldowns[skill_name] = cd
                    
                    # 특정 스킬 식별
                    if "불릿" in skill_name or "attack" in skill_name.lower():
                        self.attack_key = key
                        print(f"[DEBUG] 기본 공격 키: {key}")
                    elif "jump" in skill_name.lower():
                        self.jump_key = key
                        print(f"[DEBUG] 점프 키: {key}")
            
            # 3. 설치기 정보 로드 (installs)
            installs = job_settings.get("installs", {})
            print(f"[DEBUG] installs 섹션: {len(installs)}개")
            
            for skill_name, skill_info in installs.items():
                key = skill_info.get("key")
                dur = skill_info.get("dur", 60.0)
                
                print(f"[DEBUG] 설치기 발견: {skill_name} -> 키 {key}")
                
                if key:
                    self.install_key_map[skill_name] = key
                    self.skill_key_map[skill_name] = key  # 전체 맵에도 추가
                    
                    if dur > 0:
                        self.skill_cooldowns[skill_name] = dur
            
            # 4. 레거시 mapping도 확인 (하위 호환성)
            mapping = data.get("mapping", {})
            for skill_name, skill_info in mapping.items():
                if skill_name not in self.skill_key_map:
                    key = skill_info.get("key")
                    if key:
                        self.skill_key_map[skill_name] = key
            
            print(f"✅ [Agent] 키 설정 로드 완료 (직업: {job_name})")
            print(f"   - 전체 스킬: {len(self.skill_key_map)}개")
            print(f"   - 설치기: {self.install_key_map}")
            print(f"   - 기본 공격: {self.attack_key}")
            print(f"   - 점프: {self.jump_key}")
            
        except FileNotFoundError:
            print(f"⚠️ [Agent] hunter_config.json 파일을 찾을 수 없습니다!")
            self.install_key_map = {"fountain": "4", "janus": "5"}
            self.attack_key = "r"
            self.jump_key = "e"
        except Exception as e:
            print(f"⚠️ [Agent] 설정 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            # 폴백 기본값
            self.install_key_map = {"fountain": "4", "janus": "5"}
            self.attack_key = "r"
            self.jump_key = "e"

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

    def _register_installs_to_navigator(self, installs_dict):
        """Config에서 로드한 설치기 정보를 Navigator에 등록"""
        try:
            from modules.navigator import InstallSkill
            
            # 기존 설치기 초기화
            self.navigator.patrol.install_skills = []
            
            # 각 설치기를 InstallSkill 객체로 변환하여 등록
            for skill_name, skill_info in installs_dict.items():
                install_skill = InstallSkill(
                    name=skill_name,
                    up=skill_info.get('up', 20),
                    down=skill_info.get('down', 8),
                    left=skill_info.get('left', 18),
                    right=skill_info.get('right', 18),
                    duration=skill_info.get('dur', 60.0)
                )
                self.navigator.patrol.install_skills.append(install_skill)
            
            print(f"✅ [Agent] Navigator에 설치기 {len(self.navigator.patrol.install_skills)}개 등록 완료")
        except Exception as e:
            print(f"⚠️ [Agent] Navigator 설치기 등록 실패: {e}")
            import traceback
            traceback.print_exc()
    
    
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
        if time.time() < self.hold_key_until or time.time() < self.busy_until:
            self.last_pos_time = time.time()
            self.stuck_count = 0
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

    def get_action(self, px, py, entropy, pid, ult_ready, sub_ready, 
                   dist_left=0, dist_right=0, dist_up=0, dist_down=0,
                   corner_tl=0, corner_tr=0, corner_bl=0, corner_br=0,
                   current_kill_count=0, job_id=None, vision=None, frame=None):
        
        # [핵심 수정] 키 설정이 없을 경우를 대비한 안전장치
        if not self.skill_key_map:
            self.load_keys_from_config()
        
        now = time.time()
        dx = px - self.prev_px if self.prev_px != -1 else 0
        dy = py - self.prev_py if self.prev_py != -1 else 0
        self.prev_px, self.prev_py = px, py

        # Kill Count Update
        if current_kill_count > self.last_kill_count: 
            self.gen_manager.update_kill()
            self.last_kill_count = current_kill_count

        # ---------------------------------------------------------
        # [1. State Checks - High Priority Blockers]
        # ---------------------------------------------------------

        # A. Busy Lock
        if now < self.busy_until: 
            return "None", f"⏳ Wait ({self.busy_until - now:.1f}s)"

        # B. Hold Key Lock
        if now < self.hold_key_until: 
            # [수정] 설치기 이름 추적하여 올바른 키 반환
            # 마지막으로 사용한 설치기 키를 반환 (첫 번째 설치기 키 사용)
            if self.install_key_map:
                first_key = list(self.install_key_map.values())[0]
                return first_key, f"🔒 Holding..."
            return "q", "🔒 Holding..."  # 폴백

        # C. Action Lock (Anti-jitter)
        if now < self.action_lock_until: 
            return self.last_action_name, "🔒 Action Locked"

        # ---------------------------------------------------------
        # [2. Movement State Management]
        # ---------------------------------------------------------

        # D. Landing Logic
        if self.waiting_for_landing:
            elapsed = now - self.jump_start_time
            if elapsed > 2.0: 
                 self.waiting_for_landing = False
                 self.landing_stable_frames = 0
            
            prev_y = self.history[-1][self.feature_cols.index('player_y')] if len(self.history) > 0 else py
            if abs(py - prev_y) < 2: 
                 self.landing_stable_frames += 1
                 if self.landing_stable_frames >= 3: 
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
                # sklearn의 feature name 경고 억제
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                
                # [핵심 수정] DataFrame 없이 numpy array로 직접 생성
                # FEATURE_COLS 순서와 정확히 일치하도록 배열 생성
                # ['player_x', 'player_y', 'delta_x', 'delta_y', 'entropy', 'platform_id', 
                #  'ult_ready', 'sub_ready', 'inv_dist_up', 'inv_dist_down', 
                #  'inv_dist_left', 'inv_dist_right', 'corner_tl', 'corner_tr', 'corner_bl', 'corner_br']
                
                feature_array = np.array([[
                    px,           # player_x (index 0)
                    py,           # player_y (index 1)
                    dx,           # delta_x (index 2)
                    dy,           # delta_y (index 3)
                    entropy,      # entropy (index 4)
                    pid,          # platform_id (index 5)
                    ult_ready,    # ult_ready (index 6)
                    sub_ready,    # sub_ready (index 7)
                    dist_up,      # inv_dist_up (index 8)
                    dist_down,    # inv_dist_down (index 9)
                    dist_left,    # inv_dist_left (index 10)
                    dist_right,   # inv_dist_right (index 11)
                    corner_tl,    # corner_tl (index 12)
                    corner_tr,    # corner_tr (index 13)
                    corner_bl,    # corner_bl (index 14)
                    corner_br,    # corner_br (index 15)
                    dist_left,    # dist_left (index 16) - 학습 시 중복 포함됨
                    dist_right    # dist_right (index 17) - 학습 시 중복 포함됨
                ]], dtype=np.float64)
                
                # numpy array를 직접 scaler.transform에 전달 (feature name 문제 없음)
                feats_scaled = self.scaler.transform(feature_array)
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
                            act = str(act).lower()  # 문자열로 명시적 변환
                            if act not in ['left', 'right', 'up', 'down', 'none', 'idle', 'jump', 'double_jump']:
                                lstm_attack_cmd = act; break
                        lstm_raw_cmd = str(self.encoder.inverse_transform([np.argmax(probs)])[0])
            except Exception as e:
                # 디버깅을 위해 예외 출력
                print(f"⚠️ [LSTM Inference Error]: {e}")
                import traceback
                traceback.print_exc()
                pass


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
        
        # [Install Logic - 핵심 수정]
        if "Positioned for Install" in nav_msg and sub_ready == 1:
            skill_name = "fountain"  # 기본값
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
            self.busy_until = now + 1.0
            
            # [핵심 수정] install_key_map에서 스킬명으로 키 조회
            install_key = self.install_key_map.get(skill_name)
            
            # 못 찾으면 전체 스킬 맵에서 조회
            if not install_key:
                install_key = self.skill_key_map.get(skill_name, "q")
            
            return install_key, f"Deploy {skill_name} (Hold {press_duration:.2f}s)"

        # 1. Movement Selection
        if nav_action != "None" and nav_action:
            # nav_action이 문자열인지 확인
            if isinstance(nav_action, str):
                final_action.append(nav_action)
                # Double Jump Logic
                if "jump" not in nav_action and random.random() < 0.6: 
                    final_action.append("double_jump")
        elif lstm_raw_cmd in ['left', 'right', 'up', 'down']:
            final_action.append(lstm_raw_cmd)
        
        # 2. Attack Selection
        attack_candidate = None
        if lstm_attack_cmd != "None" and lstm_attack_cmd:
            # LSTM이 반환한 공격 명령어에서 방향키 제거
            attack_candidate = str(lstm_attack_cmd).replace('left', '').replace('right', '').replace('up', '').replace('down', '').replace('+', '').strip()
            if not attack_candidate:  # 빈 문자열이면 None으로
                attack_candidate = None
        
        # LSTM이 공격키를 제안하지 않았으면 60% 확률로 기본 공격
        if not attack_candidate and random.random() < 0.6:
            attack_candidate = self.attack_key

        if attack_candidate:
            final_action.append(attack_candidate)
            final_msg += f" + ATK({attack_candidate})"

        if not final_action: return "None", "Idle"
        
        result_action = "+".join(final_action)
        
        # ---------------------------------------------------------
        # [5. Apply Action Lock]
        # ---------------------------------------------------------
        if result_action != self.last_action_name:
            if "left" in result_action or "right" in result_action:
                self.action_lock_until = now + 0.12
            elif "jump" in result_action:
                self.action_lock_until = now + 0.08
            
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