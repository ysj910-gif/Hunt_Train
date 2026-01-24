# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel, simpledialog 
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
from modules.job_manager import JobManager
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
        self.job_mgr = JobManager() 
        
        # Agent 먼저 초기화
        self.agent = BotAgent() 

        self.rune_manager = RuneManager()
        
        physics_file = "physics_hybrid_model.pth"
        if self.rune_manager.load_physics(physics_file):
            print(f"✅ 룬 이동용 물리 엔진({physics_file})이 로드되었습니다.")
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
        self.install_rows = []  # [신규] 설치기 설정 UI 행 관리용
        self.key_to_skill_map = {} 
        self.map_offset_x = 0
        self.map_offset_y = 0
        self.map_min_x = 0
        self.map_max_x = 1366

        # 경로 변수
        self.cur_map_path = ""
        self.cur_lstm_path = ""
        self.cur_rf_path = ""
        
        # 2. UI 구성 (한 번만 호출!)
        self.setup_ui()
        
        # 3. 설정 로드 (한 번만 호출!)
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
        
        # 봇 상태 표시
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
        tab_control.add(tab_map, text='Map & AI Model')
        tab_control.pack(expand=1, fill="both")

        # --- [Tab 1: Skills] ---
        
        # [★복구됨] 직업 정보를 담을 프레임 생성 (이게 없어서 에러가 났음)
        job_frame = ttk.LabelFrame(tab_skill, text="Player Info")
        job_frame.pack(fill="x", pady=5)
        ttk.Label(job_frame, text="Job Class:").pack(side="left", padx=5)

        # 콤보박스 생성
        job_list = self.job_mgr.get_all_jobs()
        if not job_list: job_list = ["Kinesis"]
        
        self.entry_job = ttk.Combobox(job_frame, values=job_list, state="readonly")
        self.entry_job.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_add_job = ttk.Button(job_frame, text="+", width=3, command=self.add_custom_job_action)
        self.btn_add_job.pack(side="left", padx=2)
        
        if job_list:
            self.entry_job.current(0)

        # 이벤트 연결
        self.entry_job.bind("<<ComboboxSelected>>", self.on_job_change)

        # === [구역 1] 일반 스킬 설정 ===
        setting_frame = ttk.LabelFrame(tab_skill, text="Custom Skills (Buff/Attack)")
        setting_frame.pack(fill="both", expand=True, pady=2)

        canvas_scroll = tk.Canvas(setting_frame, height=150) # 높이 조절
        scrollbar = ttk.Scrollbar(setting_frame, orient="vertical", command=canvas_scroll.yview)
        self.skill_list_frame = ttk.Frame(canvas_scroll)

        frame_id = canvas_scroll.create_window((0, 0), window=self.skill_list_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 일반 스킬 헤더
        h_frame = ttk.Frame(self.skill_list_frame)
        h_frame.pack(fill="x", pady=2)
        ttk.Label(h_frame, text="Name", width=12).pack(side="left", padx=1)
        ttk.Label(h_frame, text="Key", width=5).pack(side="left", padx=1)
        ttk.Label(h_frame, text="CD(s)", width=5).pack(side="left", padx=1)
        ttk.Button(h_frame, text="+", width=3, command=self.add_skill_row).pack(side="left", padx=5)

        # === [구역 2] 설치기 설정 (신규) ===
        install_frame = ttk.LabelFrame(tab_skill, text="Installation Skill (Map Coverage)")
        install_frame.pack(fill="x", pady=5)
        
        # 설치기 헤더 (Name, Key, Up, Down, Left, Right, Dur)
        ih_frame = ttk.Frame(install_frame)
        ih_frame.pack(fill="x", pady=2)
        headers = ["Name", "Key", "Up", "Down", "Left", "Right", "Dur(s)"]
        widths = [8, 5, 4, 4, 4, 4, 5]
        for t, w in zip(headers, widths):
            ttk.Label(ih_frame, text=t, width=w).pack(side="left", padx=1)
        
        # 설치기 리스트 프레임
        self.install_list_frame = ttk.Frame(install_frame)
        self.install_list_frame.pack(fill="x")
        
        ttk.Button(install_frame, text="+ Add Install Skill", command=self.add_install_row).pack(fill="x", pady=2)

        # 저장 버튼
        ttk.Button(tab_skill, text="💾 Save Config (All)", command=self.save_settings).pack(fill="x", pady=5)
        # --- [Tab 2: Map & AI Model] ---
        
        # 1. 맵 로드
        map_frame = ttk.LabelFrame(tab_map, text="1. Map Data (.json)")
        map_frame.pack(fill="x", pady=5, padx=5)
        self.lbl_map_name = ttk.Label(map_frame, text="No Map Loaded", foreground="gray")
        self.lbl_map_name.pack(pady=2)
        ttk.Button(map_frame, text="📂 Load Map JSON", command=self.open_map_file).pack(fill="x", padx=5, pady=5)

        # 2. AI 모델 로드
        model_frame = ttk.LabelFrame(tab_map, text="2. AI Models")
        model_frame.pack(fill="x", pady=5, padx=5)

        # [LSTM 섹션]
        self.lbl_model_name = ttk.Label(model_frame, text="LSTM: Not Loaded", foreground="gray")
        self.lbl_model_name.pack(pady=1)
        ttk.Button(model_frame, text="🧠 Load LSTM (.pth)", command=self.load_model_action).pack(fill="x", padx=5, pady=2)

        # [RF 섹션]
        ttk.Separator(model_frame, orient="horizontal").pack(fill="x", pady=5)
        self.lbl_rf_name = ttk.Label(model_frame, text="RF: Not Loaded", foreground="gray")
        self.lbl_rf_name.pack(pady=1)
        ttk.Button(model_frame, text="🌲 Load RF (.pkl)", command=self.load_rf_model_action).pack(fill="x", padx=5, pady=2)

        # 3. 오프셋 조절
        offset_frame = ttk.LabelFrame(tab_map, text="3. Position Offset")
        offset_frame.pack(fill="x", pady=5, padx=5)
        self.lbl_offset = ttk.Label(offset_frame, text="Offset: (0, 0)", font=("Arial", 10, "bold"))
        self.lbl_offset.pack(pady=2)

        self.lbl_map_info = ttk.Label(offset_frame, text="Map Info: Load Map First", foreground="gray")
        self.lbl_map_info.pack(pady=2)
        
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

        # 봇 가동 버튼
        self.btn_bot = ttk.Button(bottom_frame, text="🤖 AUTO HUNT (봇 가동)", command=self.toggle_botting, state="disabled")
        self.btn_bot.pack(fill="x", ipady=10, pady=5)

    # === [기능 구현] ===
    def _configure_canvas(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
            # 캔버스 너비에 맞춰 내부 프레임 너비 조정
            canvas_scroll.itemconfig(frame_id, width=event.width)

            canvas_scroll.bind("<Configure>", _configure_canvas)
    
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

    # gui.py -> toggle_botting 메서드 수정

    def toggle_botting(self):
        if not self.vision.window_found:
            messagebox.showwarning("경고", "먼저 창을 찾으세요.")
            return

        if self.is_botting:
            # [1] 봇 정지
            self.is_botting = False
            self.btn_bot.config(text="🤖 AUTO HUNT (봇 가동)")
            self.lbl_bot_status.config(text="[BOT: OFF]", foreground="gray")
            
            # [수정] self.log -> print로 변경 (GUI 로그 함수가 없으므로)
            print("🛑 봇 정지 중... 키 입력 해제 대기") 
            
            # 0.1초 뒤 키 해제
            self.root.after(100, lambda: self.input_handler.release_all())
            
        else:
            # [2] 봇 시작
            self.is_botting = True
            self.btn_bot.config(text="⏹ STOP BOT (중지)", state="normal")
            self.lbl_bot_status.config(text="[BOT: ON]", foreground="red")
            
            self.agent.reset_history()
            
            # 스레드 재시작 로직
            if not hasattr(self, 'bot_thread') or not self.bot_thread.is_alive():
                self.bot_thread = threading.Thread(target=self.loop)
                self.bot_thread.daemon = True
                self.bot_thread.start()
                print("🚀 봇 스레드 시작됨.") # 여기도 print로 변경

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
    
    def on_job_change(self, event=None):
        """직업 변경 시: 현재 스킬 저장 -> 새 직업 스킬 로드"""
        new_job = self.entry_job.get()
        
        # 1. (중요) 바뀌기 전 직업이 무엇이었는지 확인 필요
        # 이를 위해 self.current_job 변수를 하나 만들어 관리해야 함
        if hasattr(self, 'last_selected_job'):
            prev_job = self.last_selected_job
            print(f"💾 직업 변경: {prev_job} 설정 자동 저장 중...")
            self.save_settings(job_name_override=prev_job) # 이전 직업 강제 저장
            
        # 2. 새 직업 설정 불러오기
        print(f"📂 직업 로드: {new_job} 설정 불러오는 중...")
        self.load_job_settings(new_job)
        
        # 3. 현재 직업 갱신
        self.last_selected_job = new_job

    def add_custom_job_action(self):
        """팝업창을 띄워 새 직업 이름을 입력받고 목록에 추가"""
        new_job = simpledialog.askstring("직업 추가", "새 직업(클래스) 이름을 입력하세요:")
        
        if new_job:
            new_job = new_job.strip() # 공백 제거
            if not new_job: return

            # 현재 콤보박스에 있는 목록 가져오기
            current_values = list(self.entry_job['values'])
            
            # 중복 확인
            if new_job in current_values:
                messagebox.showwarning("중복", f"'{new_job}'은(는) 이미 있습니다.")
                self.entry_job.set(new_job)
                return

            # 1. 목록에 추가
            current_values.append(new_job)
            self.entry_job['values'] = current_values
            
            # 2. 선택된 직업을 새 직업으로 변경
            self.entry_job.set(new_job)
            
            # 3. 변경사항 반영 (스킬창 초기화 등)
            self.on_job_change()
            
            # 4. 저장 (config.json에 custom_job_list 항목으로 저장해두어야 다음에 켜도 유지됨)
            self.save_settings()
            messagebox.showinfo("완료", f"새 직업 '{new_job}'이 추가되었습니다.\n스킬을 설정하고 저장하세요.")

    def loop(self):
        """메인 루프: 진단 정보(HUD) 수집 및 봇 로그 기록 기능 추가"""
        WALL_MARGIN = 7  # 벽 감지 범위 확대
        
        while True:
            # 변수 초기화 (에러 방지)
            minimap_img = None 

            # 1. 화면 인식
            if self.vision.window_found:
                frame, entropy, kill_count, px, py = self.vision.capture_and_analyze()

                self.vision.scan_skill_status(frame)
                
                # [신규] 설치기 성공 확인 로직 (Vision 연동)
                # 봇이 "설치 확인 중" 상태이고, 타겟 스킬이 있다면
                if hasattr(self.agent.navigator.patrol, 'current_installing_skill'):
                    target_skill = self.agent.navigator.patrol.current_installing_skill
                    
                    if self.agent.busy_until > time.time() and target_skill:
                        # 해당 스킬의 아이콘이 쿨타임(어두움) 상태인지 확인
                        if self.vision.is_skill_on_cooldown(target_skill, frame):
                            print(f"✨ [Vision] {target_skill} 설치 성공 확인! 대기 해제")
                            self.agent.busy_until = 0 
                            self.agent.navigator.patrol.current_installing_skill = None # 초기화

                # [★복구] 미니맵 이미지 추출 (룬 탐색용)
                if self.vision.minimap_roi and frame is not None:
                    mx, my, mw, mh = self.vision.minimap_roi
                    # 배열 범위 안전장치
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
                current_job_name = self.entry_job.get()
            
                # 이름("Kinesis") -> ID(0) 자동 변환
                job_id = self.job_mgr.get_job_id(current_job_name)
                
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
                        
                        # [수정] vision과 frame 인자 전달 추가
                        act, msg = self.agent.get_action(
                            px, py, entropy, pid, ult, sub, 
                            current_dist_left, current_dist_right, 
                            current_kill_count=kill_count,
                            job_id=job_id,
                            vision=self.vision,  # <--- 추가됨
                            frame=frame          # <--- 추가됨
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
            # 봇이 켜져있으면 봇의 행동, 아니면 사람이 누른 키 집합(Set)을 그대로 넘김
            if self.is_botting:
                current_keys_input = action_name
            else:
                current_keys_input = self.held_keys.copy() # Set 복사

            # 5. 데이터 녹화
            if self.is_recording and self.logger:
                current_job = self.entry_job.get()
                
                # [수정] key_map을 함께 전달하여 logger가 번역하게 함
                self.logger.log_step(
                    entropy, self.skill_manager, active_skill, current_keys_input, 
                    px, py, pid, kill_count, current_dist_left, current_dist_right,
                    job_class=current_job,
                    key_map=self.key_to_skill_map # ★ 설정된 스킬 매핑 전달
                )

            # 6. GUI 업데이트 (진단 정보 전달)
            self.root.after(0, self.update_gui, frame, entropy, action_name, kill_count, px, py, debug_info)
            time.sleep(0.033)

    # gui.py -> execute_bot_action 메서드 전체 교체

    # gui.py -> execute_bot_action 메서드 전체 교체

    def execute_bot_action(self, action_name, action_msg=""):
        """
        [수정] 커스텀 점프 키('e') 인식 및 방향 점프 타이밍 보정
        """
        # ---------------------------------------------------------
        # 1. 점프 키('jump')가 무엇인지 찾기
        # ---------------------------------------------------------
        # 기본값은 'c'지만, 사용자가 등록한 'jump' 스킬이 있으면 그 키('e')를 가져옴
        jump_key = 'c' # fallback
        
        # key_map에서 'jump'라는 이름의 키가 있는지 검색 (대소문자 무관)
        for name, key in self.input_handler.key_map.items():
            if name.lower() == 'jump':
                jump_key = key
                break
        
        # ---------------------------------------------------------
        # 2. 설치기/스킬 Holding 처리
        # ---------------------------------------------------------
        if "Holding" in action_msg or "Deploy" in action_msg: 
            target_key = None
            if action_name == "sub_attack":
                if hasattr(self.agent.navigator.patrol, 'next_skill_to_use') and \
                   self.agent.navigator.patrol.next_skill_to_use:
                    skill_name = self.agent.navigator.patrol.next_skill_to_use.name
                    target_key = self.input_handler.key_map.get(skill_name)
                    
            elif action_name in self.input_handler.key_map:
                target_key = self.input_handler.key_map[action_name]
            
            if target_key:
                # 설치 때는 이동키 간섭 없게 깔끔하게 처리
                self.input_handler.release_all_except(target_key)
                self.input_handler.hold(target_key)
            return

        # ---------------------------------------------------------
        # 3. 이동 및 일반 행동 (점프 로직 핵심)
        # ---------------------------------------------------------
        rope_key = self.input_handler.key_map.get('rope', 'v')
        
        # 커맨드 변환
        if action_name == 'up': action_name = f'up+{jump_key}'
        elif action_name == 'down': action_name = f'down+{jump_key}'
        elif action_name == 'rope': action_name = f'up+{rope_key}'

        if action_name != "None":
            # "left+jump" -> {'left', 'e'} (점프키가 e인 경우)
            # action_name에는 'jump'라는 문자열이 들어오므로, 이를 실제 키(e)로 치환해야 함
            
            parts = action_name.replace('double_jump', 'jump').split('+')
            keys_to_press = set()
            
            for p in parts:
                if p == 'jump': keys_to_press.add(jump_key) # 'jump' -> 'e'
                else: keys_to_press.add(p)
            
            move_keys = {'left', 'right', 'up', 'down'}

            # [핵심 수정] 방향키 충돌 방지
            # 왼쪽을 눌러야 하면 오른쪽은 무조건 뗀다 (반대도 마찬가지)
            if 'left' in keys_to_press: self.input_handler.release('right')
            if 'right' in keys_to_press: self.input_handler.release('left')
            if 'up' in keys_to_press: self.input_handler.release('down')
            if 'down' in keys_to_press: self.input_handler.release('up')
            
            # [Step 1] 불필요한 방향키 떼기 (관성 제어)
            for k in list(self.input_handler.held_keys):
                if k in move_keys and k not in keys_to_press:
                    self.input_handler.release(k)

            # [Step 2] 방향키 먼저 누르기 (가속 시작)
            for k in keys_to_press:
                if k in move_keys:
                    self.input_handler.hold(k)
            
            # [Step 3] 점프/공격 키 입력 (방향키 입력 후 지연 실행)
            for k in keys_to_press:
                if k not in move_keys:
                    real_k = self.input_handler.key_map.get(k, k)
                    
                    # [핵심] 점프 키('e')인 경우 딜레이를 줌
                    if real_k == jump_key:
                        # 방향키가 눌리고 0.08초 뒤에 점프를 눌러야 "앞점프"가 나감
                        # 이 시간이 너무 짧으면 제자리 점프가 됨
                        time.sleep(0.08) 
                        self.input_handler.press(real_k) 
                    else:
                        self.input_handler.press(real_k)
        else:
            self.input_handler.release_all()

    # 기존 함수들 (가독성 복구 및 버그 수정)
    def open_map_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if file_path:
            if self.brain.load_map_file(file_path):
                self.cur_map_path = file_path
                self.lbl_map_name.config(text=file_path.split("/")[-1], foreground="green")
                
                self.rune_manager.load_map(file_path)
                self.agent.on_map_change(file_path)

                # [신규] 가장 왼쪽(Min X)과 가장 낮은(Max Y) 발판 좌표 찾기
                if self.brain.footholds:
                    all_x = []
                    all_y = []
                    for (x1, y1, x2, y2) in self.brain.footholds:
                        all_x.extend([x1, x2])
                        all_y.extend([y1, y2])
                    
                    min_x = min(all_x)
                    max_y = max(all_y) # Y좌표가 클수록 아래쪽
                    
                    self.map_min_x = min_x
                    self.map_max_x = max(all_x)
                    
                    # 정보 표시 (사용자가 Offset 조절할 때 참고)
                    info_text = f"Left X: {min_x} | Bottom Y: {max_y}"
                    self.lbl_map_info.config(text=info_text, foreground="blue")
                    messagebox.showinfo("맵 로드", f"로드 완료.\n{info_text}\n이 값을 참고해 Offset을 조절하세요.")
                else:
                    self.lbl_map_info.config(text="No Footholds Found")

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

    def add_install_row(self, name="", key="", up="0", down="0", left="0", right="0", dur="0"):
        row_f = ttk.Frame(self.install_list_frame)
        row_f.pack(fill="x", pady=2)
        
        entries = []
        vals = [name, key, up, down, left, right, dur]
        widths = [8, 5, 4, 4, 4, 4, 5]
        
        for v, w in zip(vals, widths):
            e = ttk.Entry(row_f, width=w)
            e.pack(side="left", padx=1)
            e.insert(0, str(v))
            entries.append(e)
            
        # [신규] 아이콘 설정 버튼 (눈 모양)
        # 버튼을 누르면 현재 행의 이름(entries[0].get())을 가져와서 ROI 설정
        btn_icon = ttk.Button(row_f, text="👁️", width=3, 
                   command=lambda: self.open_roi_selector("skill", target_name=entries[0].get()))
        btn_icon.pack(side="left", padx=2)

        # 삭제 버튼
        ttk.Button(row_f, text="X", width=2, command=lambda: self.delete_install_row(row_f)).pack(side="left", padx=2)
        
        self.install_rows.append({
            "frame": row_f,
            "name": entries[0], "key": entries[1],
            "up": entries[2], "down": entries[3], 
            "left": entries[4], "right": entries[5],
            "dur": entries[6]
        })

    def delete_install_row(self, row_frame):
        row_frame.destroy()
        self.install_rows = [r for r in self.install_rows if r["frame"] != row_frame]

    def save_settings(self, job_name_override=None):
        target_job = job_name_override if job_name_override else self.entry_job.get()
        
        # 1. 일반 스킬 읽기
        skill_mapping = {}
        for r in self.skill_rows:
            try:
                if r["frame"].winfo_exists() and r["name"].get():
                    skill_mapping[r["name"].get()] = {
                        "key": r["key"].get(), "cd": float(r["cd"].get() or 0), "dur": float(r["dur"].get() or 0)
                    }
            except: pass
            
        # 2. [신규] 설치기 스킬 읽기
        install_mapping = {}
        for r in self.install_rows:
            try:
                if r["frame"].winfo_exists() and r["name"].get():
                    install_mapping[r["name"].get()] = {
                        "key": r["key"].get(),
                        "up": int(r["up"].get() or 0), "down": int(r["down"].get() or 0),
                        "left": int(r["left"].get() or 0), "right": int(r["right"].get() or 0),
                        "dur": float(r["dur"].get() or 0)
                    }
            except: pass

        # 3. 저장
        data = utils.load_config()
        if "job_settings" not in data: data["job_settings"] = {}
        
        # 직업별 데이터 저장 구조 개선
        data["job_settings"][target_job] = {
            "skills": skill_mapping,
            "installs": install_mapping
        }

                
        # 공통 설정 저장
        data["last_job"] = self.entry_job.get()
        data["map_offset_x"] = self.map_offset_x
        data["map_offset_y"] = self.map_offset_y

        # [수정] ROI 설정 저장 (설치기 아이콘 포함)
        if self.vision.minimap_roi:
            data["minimap_roi"] = self.vision.minimap_roi
        if self.vision.kill_roi:
            data["kill_roi"] = self.vision.kill_roi
            
        # [신규] 설치기 아이콘 ROI 저장
        if hasattr(self.vision, 'skill_rois') and self.vision.skill_rois:
            # 튜플은 JSON 저장 시 리스트로 변환됨
            data["skill_rois"] = self.vision.skill_rois
            
        # 파일 경로들 기억
        data["last_map_path"] = self.cur_map_path
        data["last_lstm_path"] = self.cur_lstm_path
        data["last_rf_path"] = self.cur_rf_path
        
        utils.save_config(data)
        
        if target_job == self.entry_job.get():
            self.update_logic_from_ui()
            messagebox.showinfo("저장됨", f"[{target_job}] 설정이 저장되었습니다.")

    def load_job_settings(self, job_name):
        data = utils.load_config()
        job_data = data.get("job_settings", {}).get(job_name, {})
        
        # 하위 호환성 (구버전 config는 딕셔너리가 바로 스킬맵)
        if "skills" in job_data or "installs" in job_data:
            skill_map = job_data.get("skills", {})
            install_map = job_data.get("installs", {})
        else:
            skill_map = job_data
            install_map = {}

        # 1. UI 비우기
        for r in self.skill_rows: r["frame"].destroy()
        self.skill_rows = []
        for r in self.install_rows: r["frame"].destroy()
        self.install_rows = []
        
        # 2. 일반 스킬 로드
        for name, info in skill_map.items():
            self.add_skill_row(name, info.get("key"), str(info.get("cd",0)), str(info.get("dur",0)))
            
        # 3. 설치기 로드
        for name, info in install_map.items():
            self.add_install_row(
                name, info.get("key"), 
                str(info.get("up",0)), str(info.get("down",0)),
                str(info.get("left",0)), str(info.get("right",0)),
                str(info.get("dur",0))
            )
            
        self.update_logic_from_ui()

    def load_settings(self):
        """앱 시작 시 설정 로드 (직업별 스킬 + 공통 설정 통합 복구)"""
        import os
        data = utils.load_config()
        
        saved_jobs = data.get("saved_job_list", [])
        if saved_jobs:
            # 기본 목록(job_manager)과 합치기 (중복 제거)
            default_jobs = self.job_mgr.get_all_jobs() if self.job_mgr.get_all_jobs() else ["Kinesis"]
            final_list = sorted(list(set(default_jobs + saved_jobs)))
            self.entry_job['values'] = final_list

        # 1. 마지막 직업 복구 및 선택
        last_job = data.get("last_job", "Kinesis")
        self.entry_job.set(last_job)
        self.last_selected_job = last_job  # 현재 직업 상태 기억
        
        # 2. 공통 설정 복구 (오프셋)
        self.map_offset_x = data.get("map_offset_x", 0)
        self.map_offset_y = data.get("map_offset_y", 0)
        self.lbl_offset.config(text=f"Offset: ({self.map_offset_x}, {self.map_offset_y})")
        
        # 3. ROI 영역 복구
        minimap_roi = data.get("minimap_roi")
        if minimap_roi: self.vision.set_minimap_roi(tuple(minimap_roi))
        
        kill_roi = data.get("kill_roi")
        if kill_roi: self.vision.set_roi(tuple(kill_roi))

        # [신규] 설치기 아이콘 ROI 복구
        saved_skill_rois = data.get("skill_rois", {})
        if saved_skill_rois:
            for s_name, s_data in saved_skill_rois.items():
                rect = tuple(s_data['rect'])
                thresh = s_data['threshold']
                # 프레임 없이 저장된 값으로 복구
                self.vision.set_skill_roi(s_name, rect, threshold=thresh)
            print(f"✅ 저장된 설치기 ROI {len(saved_skill_rois)}개 복구 완료")

        # 4. 파일 경로 및 모델 복구
        # 4-1. 맵 파일 (.json)
        map_path = data.get("last_map_path", "")
        if map_path and os.path.exists(map_path):
            if self.brain.load_map_file(map_path):
                self.cur_map_path = map_path
                self.lbl_map_name.config(text=map_path.split("/")[-1], foreground="green")
                
                # [중요] 맵 변경 사항을 룬 매니저와 Agent에게도 전파
                self.rune_manager.load_map(map_path)
                self.agent.on_map_change(map_path)

        # 4-2. LSTM 모델 (.pth)
        lstm_path = data.get("last_lstm_path", "")
        if lstm_path and os.path.exists(lstm_path):
            success, _ = self.agent.load_lstm(lstm_path)
            if success:
                self.cur_lstm_path = lstm_path
                self.lbl_model_name.config(text=f"LSTM: {lstm_path.split('/')[-1]}", foreground="blue")
                self.btn_bot.config(state="normal")

        # 4-3. RF 모델 (.pkl)
        rf_path = data.get("last_rf_path", "")
        if rf_path and os.path.exists(rf_path):
            success, _ = self.agent.load_rf(rf_path)
            if success:
                self.cur_rf_path = rf_path
                self.lbl_rf_name.config(text=f"RF: {rf_path.split('/')[-1]}", foreground="green")

        # 5. [핵심] 해당 직업의 스킬 세팅 로드
        # 이제 직접 매핑을 읽지 않고, 직업별 로더에게 위임합니다.
        self.load_job_settings(last_job)

    def update_logic_from_ui(self):
        # 1. 일반 스킬 업데이트
        self.key_to_skill_map.clear()
        new_cd = {}; new_dur = {}; new_km = {}
        
        for r in self.skill_rows:
            try:
                name = r["name"].get(); key = r["key"].get().lower()
                if name and key:
                    new_cd[name] = float(r["cd"].get() or 0)
                    new_dur[name] = float(r["dur"].get() or 0)
                    self.key_to_skill_map[key] = name; new_km[name] = key
            except: pass
            
        # 2. [수정] 설치기 업데이트 (모든 행을 리스트로 전달)
        if hasattr(self.agent, 'navigator') and hasattr(self.agent.navigator, 'patrol'):
            from modules.navigator import InstallSkill
            
            # 기존 리스트 초기화
            self.agent.navigator.patrol.install_skills = []
            
            for r in self.install_rows:
                try:
                    i_name = r["name"].get()
                    i_key = r["key"].get().lower()
                    
                    if i_name and i_key:
                        # 키 매핑 등록
                        self.key_to_skill_map[i_key] = i_name
                        new_km[i_name] = i_key
                        
                        # 지속시간
                        i_dur = float(r["dur"].get() or 60.0)
                        # 설치기는 쿨타임 = 지속시간으로 관리
                        new_cd[i_name] = i_dur 
                        
                        # 범위 값
                        up = int(r["up"].get() or 0)
                        down = int(r["down"].get() or 0)
                        left = int(r["left"].get() or 0)
                        right = int(r["right"].get() or 0)
                        
                        # 객체 생성 및 리스트에 추가
                        new_skill = InstallSkill(i_name, up, down, left, right, i_dur)
                        self.agent.navigator.patrol.install_skills.append(new_skill)
                        
                except Exception as e:
                    print(f"Error parsing install row: {e}")
                    
            print(f"🛠️ [GUI] 설치기 {len(self.agent.navigator.patrol.install_skills)}개 업데이트 완료")

        self.skill_manager.update_skill_list(new_cd, new_dur)
        self.input_handler.update_key_map(new_km)
        
        # HUD 쿨타임 바 재생성
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

    # gui.py -> update_gui 함수 전체 교체

    def update_gui(self, frame, entropy, action, kill, px, py, debug_info):
        if frame is not None:
            # 1. 발판 그리기 (JSON 데이터 - 빨간선) + 미니맵 보정
            if self.brain.footholds and self.vision.minimap_roi:
                mx, my, _, _ = self.vision.minimap_roi
                for (x1, y1, x2, y2) in self.brain.footholds:
                    draw_x1 = x1 + self.map_offset_x + mx
                    draw_y1 = y1 + self.map_offset_y + my
                    draw_x2 = x2 + self.map_offset_x + mx
                    draw_y2 = y2 + self.map_offset_y + my
                    
                    if 0 <= draw_x1 < frame.shape[1] and 0 <= draw_y1 < frame.shape[0]:
                        cv2.line(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 0, 255), 2)
            
            # 2. [수정] 활성화된 설치기 영역 그리기 (Cyan Box) & 텍스트 표시
            active_install_texts = [] # 화면에 띄울 텍스트 리스트
            
            if hasattr(self.agent, 'navigator') and hasattr(self.agent.navigator, 'patrol'):
                patrol = self.agent.navigator.patrol
                
                if hasattr(patrol, 'active_installs'):
                    for ins in patrol.active_installs:
                        ix, iy = ins['pos']
                        skill = ins['skill']
                        rem_time = ins['expiry'] - time.time()
                        
                        # 텍스트 정보 수집
                        active_install_texts.append(f"📍 {skill.name}: {rem_time:.1f}s left")

                        # 미니맵 기준 좌표 변환
                        base_x = ix + self.map_offset_x
                        base_y = iy + self.map_offset_y
                        if self.vision.minimap_roi:
                            base_x += self.vision.minimap_roi[0]
                            base_y += self.vision.minimap_roi[1]
                        
                        # 박스 그리기
                        x1 = int(base_x - skill.real_range['left'])
                        x2 = int(base_x + skill.real_range['right'])
                        y1 = int(base_y - skill.real_range['up'])
                        y2 = int(base_y + skill.real_range['down'])
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f"{skill.name}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # 3. 캐릭터 위치 (초록 원)
            if self.vision.minimap_roi and px > 0:
                mx, my, _, _ = self.vision.minimap_roi
                cv2.circle(frame, (mx+px, my+py), 5, (0, 255, 0), -1)

            # 4. Platform ID 디버깅 (왼쪽 하단 유지)
            pid = self.find_platform_id(px, py)
            color = (0, 255, 0) if pid != -1 else (0, 0, 255)
            cv2.putText(frame, f"Plat ID: {pid}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ---------------------------------------------------------
            # [HUD 오버레이] 오른쪽으로 이동 (X=350)
            # ---------------------------------------------------------
            HUD_X = 350  # 요청하신 대로 오른쪽으로 이동
            y_pos = 40
            
            # A. 봇 상태 정보
            cv2.putText(frame, f"ACT: {action}", (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 25
            
            cycle = debug_info.get("Cycle", "OFF")
            c_color = (0, 0, 255) if cycle == "COMBAT" else (255, 0, 0)
            cv2.putText(frame, f"MODE: {cycle}", (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)
            y_pos += 25
            
            nav = debug_info.get("Nav", "")
            cv2.putText(frame, f"MSG: {nav}", (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_pos += 25
            
            stuck = debug_info.get("Stuck", "0")
            if stuck != "0/2":
                cv2.putText(frame, f"STUCK: {stuck}", (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_pos += 25

            # B. 스킬 인식 정보 (Value / Threshold)
            y_pos += 10 
            if hasattr(self.vision, 'skill_debug_info') and self.vision.skill_debug_info:
                for name, info in self.vision.skill_debug_info.items():
                    val = info['val']   # 현재 밝기
                    thr = info['thr']   # 기준 밝기
                    is_cool = info['is_cool'] # 쿨타임 여부
                    
                    status_str = "[COOL]" if is_cool else "[READY]"
                    s_color = (0, 0, 255) if is_cool else (0, 255, 0) # 쿨타임=빨강(성공), 대기=초록
                    
                    # 텍스트: "Fountain: 135.0 < 150.0 [COOL]"
                    text = f"{name}: {val:.1f} < {thr:.1f} {status_str}"
                    cv2.putText(frame, text, (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)
                    y_pos += 25
            else:
                cv2.putText(frame, "⚠️ No Skill ROI Set", (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_pos += 25

            # C. 활성화된 설치기 목록 (Active List)
            y_pos += 10
            if active_install_texts:
                cv2.putText(frame, "=== Active Installs ===", (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 25
                for txt in active_install_texts:
                    cv2.putText(frame, txt, (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_pos += 25
            else:
                cv2.putText(frame, "No Active Installs", (HUD_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

            # 이미지 변환 및 출력
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)))
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

    def open_roi_selector(self, target, target_name=None):
        if not self.vision.window_found: return
        self.roi_target = target
        self.roi_target_name = target_name # [신규] 어떤 스킬인지 기억
        
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
            frame, _, _, _, _ = self.vision.capture_and_analyze()
            
            if self.roi_target == "kill": self.vision.set_roi(rect)
            elif self.roi_target == "minimap": self.vision.set_minimap_roi(rect)
            elif self.roi_target == "skill": 
                # [신규] 이름과 함께 등록
                if self.roi_target_name:
                    self.vision.set_skill_roi(self.roi_target_name, rect, frame)
                    messagebox.showinfo("설정", f"[{self.roi_target_name}] 아이콘 영역 설정됨")
                else:
                    messagebox.showwarning("오류", "스킬 이름을 먼저 입력하세요.")
            
            if self.roi_target != "skill":
                messagebox.showinfo("설정", f"{self.roi_target} 영역 설정됨")
            
            win.destroy()