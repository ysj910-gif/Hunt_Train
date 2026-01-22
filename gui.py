# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel
from PIL import Image, ImageTk
import cv2
import threading
import time
import random
from pynput import keyboard

# 모듈 임포트
from modules.vision import VisionSystem
from modules.brain import SkillManager
from modules.input import InputHandler
from modules.logger import DataLogger
from modules.agent import BotAgent  # [신규] 뇌 담당 Agent
from modules.humanizer import Humanizer  # [추가]
from modules.rune_solver import RuneManager  # [추가]
import utils


class MapleHunterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maple Hunter Modular Ver.")
        self.root.geometry("1200x950")

        # 1. 핵심 모듈 초기화
        self.vision = VisionSystem()
        self.skill_manager = SkillManager()
        self.input_handler = InputHandler()
        self.humanizer = Humanizer()
        
        # Agent 먼저 초기화
        self.agent = BotAgent() 

        self.rune_manager = RuneManager()
        
        # [★수정] UI가 아직 없으므로 print로만 출력하고, lbl_physics.config 코드는 삭제함
        physics_file = "physics_hybrid_model.pth"
        if self.rune_manager.load_physics(physics_file):
            print(f"✅ 룬 이동용 물리 엔진({physics_file})이 로드되었습니다.")
            # self.lbl_physics.config(...)  <-- [삭제] 이 줄이 에러 원인이었음!
        else:
            print(f"⚠️ 물리 엔진 파일({physics_file})이 없습니다. 'train_physics.py'를 실행하세요.")
        
        # Humanizer 설정
        self.humanizer.blending_ratio = 0.7
        self.exploration_rate = 0.05
        
        # Brain (발판 정보)
        from modules.brain import StrategyBrain 
        self.brain = StrategyBrain(self.skill_manager)

        # 상태 변수
        self.is_recording = False
        self.is_botting = False
        self.held_keys = set()
        
        self.skill_rows = []
        self.key_to_skill_map = {} 
        self.map_offset_x = 0
        self.map_offset_y = 0
        self.map_min_x = 0
        self.map_max_x = 1366

        # 경로 변수
        self.cur_map_path = ""
        self.cur_lstm_path = ""
        self.cur_rf_path = ""

        # 2. UI 구성 (여기서 라벨들이 생성됨)
        self.setup_ui()
        
        # 3. 설정 로드
        self.load_settings()
        
        # 4. 백그라운드 작업
        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()
        
        threading.Thread(target=self.humanizer.fit_from_logs, daemon=True).start()
        
        # 메인 루프
        threading.Thread(target=self.loop, daemon=True).start()
        
    def on_key_press(self, key):
        # [수정] 봇이 켜져있을 땐 물리 키보드 입력을 무시함 (중복 기록 방지)
        if self.is_recording and not self.is_botting: 
            try: self.held_keys.add(self.get_key_name(key))
            except: pass

    def on_key_release(self, key):
        # [수정] 봇이 켜져있을 땐 물리 키보드 입력을 무시함
        if self.is_recording and not self.is_botting:
            try:
                k = self.get_key_name(key)
                if k in self.held_keys: self.held_keys.remove(k)
            except: pass

    def get_key_name(self, key):
        if hasattr(key, 'char') and key.char: return key.char.lower()
        else: return str(key).replace("Key.", "")
    
    def setup_ui(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)
        
        left = ttk.Frame(paned, padding=10)
        right = ttk.Frame(paned, padding=10)
        paned.add(left, weight=2)
        paned.add(right, weight=1)

        # === [Left] 화면 및 상태 ===
        self.canvas = tk.Canvas(left, bg="black", height=360)
        self.canvas.pack(fill="x", pady=5)
        
        status_frame = ttk.Frame(left)
        status_frame.pack(fill="x", pady=10)
        
        self.lbl_entropy = ttk.Label(status_frame, text="Entropy: 0", font=("Consolas", 12), foreground="blue")
        self.lbl_entropy.pack(side="left", padx=5)
        self.lbl_kill = ttk.Label(status_frame, text="Kills: 0", font=("Consolas", 12, "bold"), foreground="green")
        self.lbl_kill.pack(side="left", padx=15)
        
        # 봇 상태 표시 (추가됨)
        self.lbl_bot_status = ttk.Label(status_frame, text="[BOT: OFF]", font=("Consolas", 14, "bold"), foreground="gray")
        self.lbl_bot_status.pack(side="right", padx=5)
        self.lbl_action = ttk.Label(status_frame, text="Act: None", font=("Consolas", 14, "bold"), foreground="red")
        self.lbl_action.pack(side="right", padx=15)

        self.cooldown_frame = ttk.Frame(left)
        self.cooldown_frame.pack(fill="x", pady=5)

        # === [Right] 설정 탭 ===
        tab_control = ttk.Notebook(right)
        tab_skill = ttk.Frame(tab_control)
        tab_map = ttk.Frame(tab_control) 
        
        tab_control.add(tab_skill, text='Skills & Info')
        tab_control.add(tab_map, text='Map & AI Model') # 이름 변경
        tab_control.pack(expand=1, fill="both")

        # --- [Tab 1: Skills] ---
        job_frame = ttk.LabelFrame(tab_skill, text="Player Info")
        job_frame.pack(fill="x", pady=5)
        ttk.Label(job_frame, text="Job Class:").pack(side="left", padx=5)
        self.entry_job = ttk.Entry(job_frame)
        self.entry_job.pack(side="left", fill="x", expand=True, padx=5)

        setting_frame = ttk.LabelFrame(tab_skill, text="Custom Skills")
        setting_frame.pack(fill="both", expand=True, pady=5)

        canvas_scroll = tk.Canvas(setting_frame, height=250)
        scrollbar = ttk.Scrollbar(setting_frame, orient="vertical", command=canvas_scroll.yview)
        self.skill_list_frame = ttk.Frame(canvas_scroll)

        self.skill_list_frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=self.skill_list_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 헤더
        h_frame = ttk.Frame(self.skill_list_frame)
        h_frame.pack(fill="x", pady=2)
        ttk.Label(h_frame, text="Skill Name", width=15).pack(side="left", padx=2)
        ttk.Label(h_frame, text="Key", width=6).pack(side="left", padx=2)
        ttk.Label(h_frame, text="CD(s)", width=6).pack(side="left", padx=2)

        skill_btn_frame = ttk.Frame(tab_skill)
        skill_btn_frame.pack(fill="x", pady=5)
        ttk.Button(skill_btn_frame, text="+ Add Skill", command=self.add_skill_row).pack(fill="x", pady=2)
        ttk.Button(skill_btn_frame, text="💾 Save Config", command=self.save_settings).pack(fill="x", pady=5)

        # --- [Tab 2: Map & AI Model] ---
        
        # 1. 맵 로드
        map_frame = ttk.LabelFrame(tab_map, text="1. Map Data (.json)")
        map_frame.pack(fill="x", pady=5, padx=5)
        self.lbl_map_name = ttk.Label(map_frame, text="No Map Loaded", foreground="gray")
        self.lbl_map_name.pack(pady=2)
        ttk.Button(map_frame, text="📂 Load Map JSON", command=self.open_map_file).pack(fill="x", padx=5, pady=5)

        # 2. AI 모델 로드 (신규 기능)
        
        model_frame = ttk.LabelFrame(tab_map, text="2. AI Models")
        model_frame.pack(fill="x", pady=5, padx=5)

        # [LSTM 섹션]
        self.lbl_model_name = ttk.Label(model_frame, text="LSTM: Not Loaded", foreground="gray")
        self.lbl_model_name.pack(pady=1)
        ttk.Button(model_frame, text="🧠 Load LSTM (.pth)", command=self.load_model_action).pack(fill="x", padx=5, pady=2)

        # [RF 섹션 - 새로 추가됨]
        ttk.Separator(model_frame, orient="horizontal").pack(fill="x", pady=5) # 구분선
        self.lbl_rf_name = ttk.Label(model_frame, text="RF: Not Loaded", foreground="gray") # 라벨 초기화 (필수)
        self.lbl_rf_name.pack(pady=1)
        ttk.Button(model_frame, text="🌲 Load RF (.pkl)", command=self.load_rf_model_action).pack(fill="x", padx=5, pady=2)

        # 3. 오프셋 조절
        offset_frame = ttk.LabelFrame(tab_map, text="3. Position Offset")
        offset_frame.pack(fill="x", pady=5, padx=5)
        self.lbl_offset = ttk.Label(offset_frame, text="Offset: (0, 0)", font=("Arial", 10, "bold"))
        self.lbl_offset.pack(pady=2)
        
        btn_pad = ttk.Frame(offset_frame)
        btn_pad.pack(pady=2)
        ttk.Button(btn_pad, text="▲", width=3, command=lambda: self.adjust_offset(0, -1)).grid(row=0, column=1)
        ttk.Button(btn_pad, text="◀", width=3, command=lambda: self.adjust_offset(-1, 0)).grid(row=1, column=0)
        ttk.Button(btn_pad, text="▼", width=3, command=lambda: self.adjust_offset(0, 1)).grid(row=1, column=1)
        ttk.Button(btn_pad, text="▶", width=3, command=lambda: self.adjust_offset(1, 0)).grid(row=1, column=2)
        ttk.Button(offset_frame, text="Reset", command=lambda: self.adjust_offset(0, 0, reset=True)).pack(pady=2)

        ttk.Separator(model_frame, orient="horizontal").pack(fill="x", pady=5)
        self.lbl_physics = ttk.Label(model_frame, text="Physics: Auto-Loaded", foreground="gray")
        self.lbl_physics.pack(pady=1)
        ttk.Button(model_frame, text="🔄 Reload Physics JSON", command=self.reload_physics_action).pack(fill="x", padx=5, pady=2)

        # --- [Bottom Controls] ---
        bottom_frame = ttk.Frame(right)
        bottom_frame.pack(side="bottom", fill="x", pady=10)

        self.btn_find_win = ttk.Button(bottom_frame, text="🔍 메이플 창 찾기", command=self.find_window_action)
        self.btn_find_win.pack(fill="x", pady=2)
        
        # 영역 설정 버튼들
        roi_frame = ttk.Frame(bottom_frame)
        roi_frame.pack(fill="x", pady=2)
        ttk.Button(roi_frame, text="🎯 킬 카운트 영역", command=lambda: self.open_roi_selector("kill")).pack(side="left", fill="x", expand=True, padx=1)
        ttk.Button(roi_frame, text="🗺️ 미니맵 영역", command=lambda: self.open_roi_selector("minimap")).pack(side="right", fill="x", expand=True, padx=1)

        # 녹화 버튼
        self.btn_record = ttk.Button(bottom_frame, text="⏺ REC (데이터 녹화)", command=self.toggle_recording)
        self.btn_record.pack(fill="x", ipady=5, pady=5)

        # [신규] 봇 가동 버튼
        self.btn_bot = ttk.Button(bottom_frame, text="🤖 AUTO HUNT (봇 가동)", command=self.toggle_botting, state="disabled")
        self.btn_bot.pack(fill="x", ipady=10, pady=5)

    # === [기능 구현] ===

    def load_model_action(self):
        path = filedialog.askopenfilename(title="Select LSTM .pth", filetypes=[("PyTorch Model", "*.pth")])
        if path:
            success, msg = self.agent.load_lstm(path)
            if success:
                self.cur_lstm_path = path # [★추가] 경로 기억
                self.lbl_model_name.config(text=f"LSTM: {path.split('/')[-1]}", foreground="blue")
                self.btn_bot.config(state="normal")
                messagebox.showinfo("로드 성공", msg)
            else:
                messagebox.showerror("로드 실패", msg)

    def load_rf_model_action(self):
        path = filedialog.askopenfilename(title="Select RF .pkl", filetypes=[("Pickle files", "*.pkl")])
        if path:
            success, msg = self.agent.load_rf(path)
            if success:
                self.cur_rf_path = path # [★추가] 경로 기억
                self.lbl_rf_name.config(text=f"RF: {path.split('/')[-1]}", foreground="green")
                messagebox.showinfo("로드 성공", msg)
            else:
                messagebox.showerror("로드 실패", msg)

    def toggle_botting(self):
        if not self.vision.window_found:
            messagebox.showwarning("경고", "먼저 창을 찾으세요.")
            return

        if self.is_botting:
            self.is_botting = False
            self.btn_bot.config(text="🤖 AUTO HUNT (봇 가동)")
            self.lbl_bot_status.config(text="[BOT: OFF]", foreground="gray")
            self.input_handler.release_all()
        else:
            self.is_botting = True
            self.btn_bot.config(text="⏹ STOP BOT (중지)", state="normal")
            self.lbl_bot_status.config(text="[BOT: ON]", foreground="red")
            
            # [수정] self.history.clear() 삭제 (BotAgent가 알아서 관리함)
            self.agent.reset_history()

    def find_platform_id(self, px, py):
        """
        [수정] 더 너그러운 판정 로직 적용
        - 시각적으로는 맞아도 좌표가 1~2픽셀 어긋날 수 있음을 보정
        """
        if not self.brain.footholds: return -1
        
        best_id = -1
        min_dist = 999  # 가장 가까운 발판을 찾기 위한 초기값
        
        # [핵심] 판정 여유 범위 (Tolerance)
        # X축: 발판 끝에서 5픽셀 정도는 벗어나도 인정
        # Y축: 발판 위아래 12픽셀 까지는 인정 (점프 중이거나 좌표 오차 고려)
        X_TOLERANCE = 3  
        Y_TOLERANCE = 5 

        for i, (x1, y1, x2, y2) in enumerate(self.brain.footholds):
            # 오프셋 적용 (화면에 그려지는 빨간 선과 동일한 좌표 계산)
            fx1 = x1 + self.map_offset_x
            fy = y1 + self.map_offset_y
            fx2 = x2 + self.map_offset_x
            
            # 1. X축 범위 확인 (여유 범위 포함)
            if (fx1 - X_TOLERANCE) <= px <= (fx2 + X_TOLERANCE):
                dist = abs(py - fy)
                
                # 2. Y축 높이 확인 (가장 가까운 발판 찾기)
                if dist < Y_TOLERANCE:
                    if dist < min_dist:
                        min_dist = dist
                        best_id = i
        
        return best_id

    def reload_physics_action(self):
        # [★핵심 수정] 파일명 변경
        if self.rune_manager.load_physics("physics_hybrid_model.pth"):
            self.lbl_physics.config(text="Physics: Loaded", foreground="green")
            messagebox.showinfo("성공", "물리 엔진을 다시 불러왔습니다.")
        else:
            messagebox.showerror("실패", "physics_hybrid_model.pth 파일이 없습니다.\ntrain_physics.py를 실행하세요.")
    
    def loop(self):
        """메인 루프: 진단 정보(HUD) 수집 및 봇 로그 기록 기능 추가"""
        WALL_MARGIN = 7  # 벽 감지 범위 확대
        
        while True:
            # 1. 화면 인식
            if self.vision.window_found:
                frame, entropy, kill_count, px, py = self.vision.capture_and_analyze()
                minimap_img = None
                if self.vision.minimap_roi and frame is not None:
                    mx, my, mw, mh = self.vision.minimap_roi
                    if 0 <= my < my+mh <= frame.shape[0] and 0 <= mx < mx+mw <= frame.shape[1]:
                        minimap_img = frame[my:my+mh, mx:mx+mw]
            else:
                frame, px, py = None, 0, 0
                time.sleep(0.5); continue

            # 2. 기본 정보 계산
            pid = self.find_platform_id(px, py)
            current_dist_left = px - self.map_min_x if px > 0 else 0
            current_dist_right = self.map_max_x - px if px > 0 else 0
            
            # 진단용 변수 초기화
            action_name = "None"
            active_skill = "Idle"
            debug_info = {} # 화면에 그릴 정보들

            # 3. 봇 로직 수행
            if self.is_botting:
                try:
                    # 3-1. 룬 탐색
                    self.rune_manager.scan_for_rune(minimap_img)
                    if self.rune_manager.rune_pos and px > 0:
                        self.agent.action_queue.clear()
                        r_act, r_msg = self.rune_manager.get_move_action(px, py)
                        if r_act:
                            if r_act == "interact": action_name = "space"; active_skill = "Rune Act"
                            else: action_name = r_act; active_skill = f"Rune: {r_msg}"

                    # 3-2. 젠 사이클 및 행동 결정 (룬이 없을 때만)
                    if action_name == "None":
                        ult = 1 if self.skill_manager.is_ready("ultimate") else 0
                        sub = 1 if self.skill_manager.is_ready("sub_attack") else 0
                        
                        # [핵심] Agent에게 킬 카운트를 넘겨줘서 젠 타이밍 계산 유도
                        act, msg = self.agent.get_action(
                            px, py, entropy, pid, ult, sub, 
                            current_dist_left, current_dist_right, 
                            current_kill_count=kill_count
                        )
                        action_name = act
                        active_skill = msg

                    # 3-3. 벽 충돌 방지 (Emergency Override)
                    if px > 0:
                        if px < self.map_min_x + WALL_MARGIN and 'left' in action_name:
                            self.agent.action_queue.clear()
                            action_name = 'right'; active_skill = "Wall(L) Fix"
                        elif px > self.map_max_x - WALL_MARGIN and 'right' in action_name:
                            self.agent.action_queue.clear()
                            action_name = 'left'; active_skill = "Wall(R) Fix"

                    # 3-4. [진단 정보 수집] 화면에 표시할 내용 정리
                    debug_info = {
                        "Cycle": self.agent.gen_manager.check_cycle(),
                        "Pattern": "Ready" if self.agent.gen_manager.pattern_queue else "Empty",
                        "Stuck": f"{self.agent.stuck_count}/2",
                        "Nav": active_skill
                    }

                    # 3-5. 키 입력 실행 (함수 분리됨)
                    self.execute_bot_action(action_name)

                except Exception as e:
                    print(f"Bot Error: {e}")
                    self.is_botting = False

            # 4. 키 상태 업데이트 (로그용)
            # 봇이 켜져있으면 봇의 행동(action_name)을 현재 입력 키로 간주
            if self.is_botting:
                current_keys_str = action_name
            else:
                # 봇이 꺼져있으면 사람이 누른 키 기록
                current_keys_str = "+".join(sorted(self.held_keys)) if self.held_keys else "None"

            # 5. 데이터 녹화 (CSV Log)
            # [수정] 봇이 작동 중일 때도 로그를 남겨서 나중에 분석 가능하게 함
            if self.is_recording and self.logger:
                self.logger.log_step(
                    entropy, self.skill_manager, active_skill, current_keys_str, 
                    px, py, pid, kill_count, current_dist_left, current_dist_right
                )

            # 6. GUI 업데이트 (진단 정보 전달)
            self.root.after(0, self.update_gui, frame, entropy, action_name, kill_count, px, py, debug_info)
            time.sleep(0.033)

    def execute_bot_action(self, action_name):
        """
        [최종 수정] 
        - 'jump': 사다리/윗점프용 (꾹 누르기)
        - 'double_jump': 플래시 점프용 (따닥 연타)
        """
        npc_key = "space"; jump_key = "space" 
        for n, k in self.input_handler.key_map.items():
            if n.lower() == "jump": jump_key = k

        # 방향키 보정
        if action_name == 'up': action_name = f'up+{jump_key}'
        elif action_name == 'down': action_name = f'down+{jump_key}'
        
        if action_name != "None":
            # 'right+double_jump+q' 같은 문자열 처리
            target_keys = set(action_name.replace('double_jump', jump_key).split('+'))
            
            # 스킬 사용
            for s_name, s_key in self.input_handler.key_map.items():
                if s_key in target_keys: self.skill_manager.use(s_name)
            
            move_keys = ['left', 'right', 'up', 'down']
            
            # 1. 안 쓰는 이동키 떼기
            for k in list(self.input_handler.held_keys):
                if k not in target_keys and k in move_keys: 
                    self.input_handler.release(k)
            
            # 2. 이동키 Hold
            for k in target_keys:
                if k in move_keys:
                    if k not in self.input_handler.held_keys:
                        self.input_handler.hold(k)
            
            # 3. 점프 로직 분기 (핵심!)
            if 'double_jump' in action_name:
                # [플래시 점프] 따닥!
                self.input_handler.press(jump_key)
                time.sleep(0.12) # 점프 사이 딜레이
                self.input_handler.press(jump_key)
                target_keys.discard(jump_key) # 아래에서 중복 입력 방지
                
            elif 'jump' in action_name:
                # [일반 점프/윗점프] 꾹~ (InputHandler.press의 0.1초 쿨타임 이용)
                # 사다리에서는 연타보다 꾹 누르는 게 유리할 수 있음
                # 여기서는 press(Tap)를 쓰되, 딜레이 없이 한 번만 입력
                self.input_handler.press(jump_key)

            # 4. 나머지 키 (공격 등)
            for k in target_keys:
                if k not in move_keys and k != jump_key:
                    self.input_handler.press(k)
        else:
            self.input_handler.release_all()

    # 기존 함수들 (가독성 복구 및 버그 수정)
    def open_map_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if file_path:
            if self.brain.load_map_file(file_path):
                self.cur_map_path = file_path
                self.lbl_map_name.config(text=file_path.split("/")[-1], foreground="green")
                
                # 룬 매니저 연동
                self.rune_manager.load_map(file_path)

                # [★추가됨] Agent(네비게이터)에게도 맵 변경 알림 -> 그래프 재생성
                self.agent.on_map_change(file_path)
                print(f"🗺️ [GUI] Agent에게 맵 정보 전달 완료")

                # 맵 경계 자동 계산 (기존 코드 유지)
                if self.brain.footholds:
                   all_xs = []
                   for (x1, y1, x2, y2) in self.brain.footholds:
                       all_xs.append(x1)
                       all_xs.append(x2)
                   self.map_min_x = min(all_xs)
                   self.map_max_x = max(all_xs)
                
                messagebox.showinfo("성공", f"맵 로드 완료\n벽 범위: {self.map_min_x} ~ {self.map_max_x}")
    
    def adjust_offset(self, dx, dy, reset=False):
        if reset: self.map_offset_x = 0; self.map_offset_y = 0
        else: self.map_offset_x += dx; self.map_offset_y += dy
        self.lbl_offset.config(text=f"Offset: ({self.map_offset_x}, {self.map_offset_y})")

    def add_skill_row(self, name="", key="", cd="0.0", dur="0.0"):
        row_f = ttk.Frame(self.skill_list_frame)
        row_f.pack(fill="x", pady=2)
        
        e_name = ttk.Entry(row_f, width=15); e_name.pack(side="left"); e_name.insert(0, name)
        e_key = ttk.Entry(row_f, width=6); e_key.pack(side="left"); e_key.insert(0, key)
        e_cd = ttk.Entry(row_f, width=6); e_cd.pack(side="left"); e_cd.insert(0, cd)
        e_dur = ttk.Entry(row_f, width=6); e_dur.pack(side="left"); e_dur.insert(0, dur)
        
        # [수정] 람다 대신 별도 함수 호출 (버그 수정)
        ttk.Button(row_f, text="X", width=3, command=lambda: self.delete_skill_row(row_f)).pack(side="left")
        
        self.skill_rows.append({"frame": row_f, "name": e_name, "key": e_key, "cd": e_cd, "dur": e_dur})

    def delete_skill_row(self, row_frame):
        """[복구] 스킬 행 삭제 및 리스트 정리"""
        row_frame.destroy()
        # 중요: 리스트에서도 해당 정보를 제거해야 저장 시 에러가 안 남
        self.skill_rows = [r for r in self.skill_rows if r["frame"] != row_frame]

    def save_settings(self):
        """설정 저장 (ROI, 키매핑, 파일 경로, 지속시간 포함)"""
        mapping = {}
        for r in self.skill_rows:
            try:
                # 빈 칸이거나 삭제된 행은 제외
                if r["frame"].winfo_exists() and r["name"].get():
                    mapping[r["name"].get()] = {
                        "key": r["key"].get(), 
                        "cd": float(r["cd"].get() or 0),
                        "dur": float(r["dur"].get() or 0)
                    }
            except: pass
            
        data = {
            "job_name": self.entry_job.get(),
            "mapping": mapping,
            "map_offset_x": self.map_offset_x,
            "map_offset_y": self.map_offset_y,
            "minimap_roi": self.vision.minimap_roi,
            
            # [수정] self.vision.roi -> self.vision.kill_roi 로 변경
            "kill_roi": self.vision.kill_roi, 
            
            "last_map_path": self.cur_map_path,
            "last_lstm_path": self.cur_lstm_path,
            "last_rf_path": self.cur_rf_path
        }
        utils.save_config(data)
        self.update_logic_from_ui()
        messagebox.showinfo("저장됨", "설정이 저장되었습니다.")

    def load_settings(self):
        """설정 불러오기 (자동 파일 로드 + 지속시간 복구)"""
        import os
        data = utils.load_config()
        
        self.entry_job.insert(0, data.get("job_name", "Adventurer"))
        self.map_offset_x = data.get("map_offset_x", 0)
        self.map_offset_y = data.get("map_offset_y", 0)
        self.lbl_offset.config(text=f"Offset: ({self.map_offset_x}, {self.map_offset_y})")
        
        # ROI 복구
        minimap_roi = data.get("minimap_roi")
        if minimap_roi: self.vision.set_minimap_roi(tuple(minimap_roi))
        kill_roi = data.get("kill_roi")
        if kill_roi: self.vision.set_roi(tuple(kill_roi))

        # 파일 경로 복구 (기존 로직 유지)
        map_path = data.get("last_map_path", "")
        if map_path and os.path.exists(map_path):
            if self.brain.load_map_file(map_path):
                self.cur_map_path = map_path
                self.lbl_map_name.config(text=map_path.split("/")[-1], foreground="green")

        lstm_path = data.get("last_lstm_path", "")
        if lstm_path and os.path.exists(lstm_path):
            success, _ = self.agent.load_lstm(lstm_path)
            if success:
                self.cur_lstm_path = lstm_path
                self.lbl_model_name.config(text=f"LSTM: {lstm_path.split('/')[-1]}", foreground="blue")
                self.btn_bot.config(state="normal")

        rf_path = data.get("last_rf_path", "")
        if rf_path and os.path.exists(rf_path):
            success, _ = self.agent.load_rf(rf_path)
            if success:
                self.cur_rf_path = rf_path
                self.lbl_rf_name.config(text=f"RF: {rf_path.split('/')[-1]}", foreground="green")

        map_path = data.get("last_map_path", "")
        if map_path and os.path.exists(map_path):
            if self.brain.load_map_file(map_path):
                self.cur_map_path = map_path
                self.lbl_map_name.config(text=map_path.split("/")[-1], foreground="green")
                
                # [★추가됨] 자동 로드 시에도 Agent에게 알림
                self.rune_manager.load_map(map_path)
                self.agent.on_map_change(map_path)

        # 스킬 매핑 복구 (NPC 키 및 지속시간 포함)
        mapping = data.get("mapping", {})
        
        # 기존 목록 초기화
        for r in self.skill_rows: r["frame"].destroy()
        self.skill_rows = []
        
        if not mapping: 
            self.add_skill_row("Genesis", "r", "30.0", "0.0")
        else:
            for s, i in mapping.items():
                self.add_skill_row(
                    s, 
                    i.get("key", ""), 
                    str(i.get("cd", 0)), 
                    str(i.get("dur", 0)) # [수정] 지속시간(dur) 불러오기 추가
                )
        self.update_logic_from_ui()

    def update_logic_from_ui(self):
        self.key_to_skill_map.clear()
        new_cd = {}; new_dur = {}; new_km = {}
        for r in self.skill_rows:
            try:
                if not r["frame"].winfo_exists(): continue
                name = r["name"].get(); key = r["key"].get().lower()
                if name:
                    new_cd[name] = float(r["cd"].get() or 0)
                    new_dur[name] = float(r["dur"].get() or 0)
                    if key: self.key_to_skill_map[key] = name; new_km[name] = key
            except: pass
            
        self.skill_manager.update_skill_list(new_cd, new_dur)
        self.input_handler.update_key_map(new_km)
        
        for w in self.cooldown_frame.winfo_children(): w.destroy()
        self.progress_bars = {}
        for s in new_cd:
            if new_cd[s] > 0:
                f = ttk.Frame(self.cooldown_frame); f.pack(fill="x")
                c = "green" if self.skill_manager.is_active(s) else "black"
                ttk.Label(f, text=s, width=10, foreground=c).pack(side="left")
                pb = ttk.Progressbar(f, length=100); pb.pack(side="right", fill="x", expand=True)
                self.progress_bars[s] = pb

    def toggle_recording(self):
        """[수정] 봇 가동 중이면 파일명에 'Bot_' 접두사 추가"""
        if self.is_recording:
            self.is_recording = False
            self.btn_record.config(text="⏺ REC (데이터 녹화)")
            if self.logger: 
                self.logger.close()
                messagebox.showinfo("완료", f"저장 완료:\n{self.logger.filepath}")
            self.logger = None
        else:
            if not self.vision.window_found: 
                messagebox.showwarning("경고", "먼저 메이플 창을 찾으세요.")
                return
            
            # [핵심] 봇 상태에 따라 파일명 결정
            prefix = "Bot" if self.is_botting else "Human"
            job = self.entry_job.get()
            filename = f"{prefix}_{job}"
            
            self.logger = DataLogger(filename)
            self.is_recording = True
            self.btn_record.config(text="⏹ STOP (저장 중...)", state="normal")

    def update_gui(self, frame, entropy, action, kill, px, py, debug_info):
        """화면에 진단용 HUD(자막) 그리기"""
        if frame is not None:
            # 1. 발판 및 캐릭터 그리기 (기존 동일)
            if self.brain.footholds:
                for (x1,y1,x2,y2) in self.brain.footholds:
                    cv2.line(frame, (x1+self.map_offset_x, y1+self.map_offset_y), 
                             (x2+self.map_offset_x, y2+self.map_offset_y), (0,0,255), 2)
            
            if self.vision.minimap_roi and px > 0:
                mx, my, _, _ = self.vision.minimap_roi
                cv2.circle(frame, (mx+px, my+py), 5, (0,255,0), -1)

            # 2. [신규] 진단 정보(HUD) 오버레이
            # 화면에 텍스트를 그려서 현재 봇의 상태를 표시합니다.
            y_pos = 30
            
            # A. 현재 수행 중인 행동
            cv2.putText(frame, f"ACT: {action}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 25
            
            # B. 젠 사이클 상태 (전투중 / 대기중 / 젠직전)
            cycle = debug_info.get("Cycle", "OFF")
            color = (0, 0, 255) if cycle == "COMBAT" else (255, 0, 0) # 전투=빨강, 대기=파랑
            cv2.putText(frame, f"MODE: {cycle}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 25
            
            # C. 네비게이터 메시지 (왜 움직이는지 이유)
            nav = debug_info.get("Nav", "")
            cv2.putText(frame, f"MSG: {nav}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_pos += 25
            
            # D. 고착 상태 (갇힘 카운트)
            stuck = debug_info.get("Stuck", "0")
            if stuck != "0/2": # 갇히기 시작하면 표시
                cv2.putText(frame, f"STUCK: {stuck}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 이미지 변환 및 캔버스 출력
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(frame, (640,360)), cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=img, anchor="nw")
            self.canvas.image = img
            
        # 하단 라벨 업데이트
        self.lbl_entropy.config(text=f"Ent: {entropy:.0f} | Pos: ({px},{py})")
        self.lbl_action.config(text=f"Act: {action}")
        self.lbl_kill.config(text=f"Kills: {kill}")
        
        # 쿨타임 바 업데이트
        for s, pb in getattr(self, 'progress_bars', {}).items():
            rem = self.skill_manager.get_remaining(s)
            tot = self.skill_manager.cooldowns.get(s, 1)
            pb['value'] = ((tot-rem)/tot)*100 if tot>0 else 100

    def find_window_action(self):
        if self.vision.find_maple_window(): messagebox.showinfo("성공", "창을 찾았습니다.")
        else: messagebox.showerror("실패", "창을 못 찾았습니다.")

    def open_roi_selector(self, target):
        if not self.vision.window_found: return
        self.roi_target = target
        frame, _, _, _, _ = self.vision.capture_and_analyze()
        if frame is None: return
        
        win = Toplevel(self.root); win.attributes('-topmost', True)
        img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cvs = tk.Canvas(win, width=img_tk.width(), height=img_tk.height(), cursor="cross")
        cvs.pack(); cvs.create_image(0,0,image=img_tk, anchor="nw"); cvs.img = img_tk
        
        cvs.bind("<ButtonPress-1>", lambda e: setattr(self, 'roi_start', (e.x, e.y)))
        cvs.bind("<B1-Motion>", lambda e: self._draw_rect(cvs, e.x, e.y))
        cvs.bind("<ButtonRelease-1>", lambda e: self._set_roi(win, e.x, e.y))

    def _draw_rect(self, cvs, x, y):
        cvs.delete("roi")
        if hasattr(self, 'roi_start'):
            cvs.create_rectangle(self.roi_start[0], self.roi_start[1], x, y, outline="red", tag="roi")

    def _set_roi(self, win, x, y):
        x0, y0 = self.roi_start
        x1, x2 = sorted([x0, x]); y1, y2 = sorted([y0, y])
        rect = (x1, y1, x2-x1, y2-y1)
        if rect[2]>5 and rect[3]>5:
            if self.roi_target == "kill": self.vision.set_roi(rect)
            else: self.vision.set_minimap_roi(rect)
            messagebox.showinfo("설정", f"{self.roi_target} 영역 설정됨: {rect}")
            win.destroy()