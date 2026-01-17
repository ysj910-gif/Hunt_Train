# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pynput import keyboard
import pandas as pd  # [추가] 데이터프레임 생성을 위해 필요

from modules.vision import VisionSystem
from modules.brain import SkillManager
from modules.input import InputHandler
from modules.logger import DataLogger
import utils
import config

# === [1] AI 모델 클래스 (학습된 모델 구조와 동일해야 함) ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        # 설정값을 저장해둠
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 저장된 설정값 사용
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class MapleHunterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maple Hunter All-in-One (Recorder & Bot)")
        self.root.geometry("1200x950")

        # 모듈 초기화
        self.vision = VisionSystem()
        self.skill_manager = SkillManager()
        self.logger = None 
        self.input_handler = InputHandler()
        
        from modules.brain import StrategyBrain 
        self.brain = StrategyBrain(self.skill_manager)

        # 상태 변수
        self.is_recording = False
        self.is_botting = False  # [신규] 봇 가동 상태
        self.held_keys = set()
        
        # AI 모델 관련 변수
        self.model = None
        self.scaler = None
        self.encoder = None
        self.history = deque(maxlen=10) # 10프레임 기억 저장소
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 스킬 및 오프셋
        self.skill_rows = []
        self.key_to_skill_map = {} 
        self.map_offset_x = 0
        self.map_offset_y = 0

        self.setup_ui()
        self.load_settings()
        
        # 키보드 리스너
        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.listener.start()

        # 메인 루프 시작
        threading.Thread(target=self.loop, daemon=True).start()

    def on_key_press(self, key):
        if self.is_recording:
            try: self.held_keys.add(self.get_key_name(key))
            except: pass

    def on_key_release(self, key):
        if self.is_recording:
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
        model_frame = ttk.LabelFrame(tab_map, text="2. AI Model (.pth)")
        model_frame.pack(fill="x", pady=5, padx=5)
        self.lbl_model_name = ttk.Label(model_frame, text="No Model Loaded", foreground="gray")
        self.lbl_model_name.pack(pady=2)
        ttk.Button(model_frame, text="🧠 Load LSTM Model", command=self.load_model_action).pack(fill="x", padx=5, pady=5)

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
        file_path = filedialog.askopenfilename(title="Select Model .pth", filetypes=[("PyTorch Model", "*.pth")])
        if not file_path: return
        
        try:
            # weights_only=False로 경고 방지
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            self.scaler = checkpoint['scaler']
            self.encoder = checkpoint['encoder']
            
            # [추가] 학습 때 사용한 컬럼 이름 불러오기 (없으면 기본값 사용)
            self.feature_cols = checkpoint.get('feature_cols', ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready'])
            
            input_size = checkpoint.get('input_size', 6)
            hidden_size = checkpoint.get('hidden_size', 128)
            num_layers = checkpoint.get('num_layers', 2)
            num_classes = checkpoint.get('num_classes', 10)

            self.model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            
            self.lbl_model_name.config(text=file_path.split("/")[-1], foreground="blue")
            self.btn_bot.config(state="normal")
            messagebox.showinfo("모델 로드", "LSTM 모델 로드 성공!\n이제 '봇 가동' 버튼을 누를 수 있습니다.")
            
        except Exception as e:
            messagebox.showerror("로드 실패", f"모델을 불러오는데 실패했습니다:\n{e}")

    def toggle_botting(self):
        """[신규] 봇 가동/중지 토글"""
        if not self.vision.window_found:
            messagebox.showwarning("경고", "먼저 메이플 창을 찾아주세요.")
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
            self.history.clear() # 기억 초기화

    def find_platform_id(self, px, py):
        """[신규] 현재 위치의 발판 ID 찾기"""
        if not self.brain.footholds: return -1
        best_id = -1; min_dist = 50
        for i, (x1, y1, x2, y2) in enumerate(self.brain.footholds):
            fx1 = x1 + self.map_offset_x; fy = y1 + self.map_offset_y; fx2 = x2 + self.map_offset_x
            if fx1 <= px <= fx2:
                dist = abs(py - fy)
                if dist < min_dist: min_dist = dist; best_id = i
        return best_id

    def loop(self):
        """메인 처리 루프 (녹화 및 봇 구동 공용)"""
        while True:
            # 1. 화면 캡처
            if self.vision.window_found:
                frame, entropy, kill_count, px, py = self.vision.capture_and_analyze()
            else:
                frame, entropy, kill_count, px, py = None, 0, 0, 0, 0
                time.sleep(0.5); continue

            # 2. 공통 데이터 계산
            current_keys_str = "+".join(sorted(self.held_keys)) if self.held_keys else "None"
            pid = self.find_platform_id(px, py)
            
            active_skill = "Idle"
            
            # --- [녹화 모드] ---
            if self.is_recording and self.logger:
                for k in self.held_keys:
                    if k in self.key_to_skill_map:
                        s_name = self.key_to_skill_map[k]
                        self.skill_manager.use(s_name)
                        active_skill = s_name
                self.logger.log_step(entropy, self.skill_manager, active_skill, current_keys_str, px, py, pid, kill_count)

            # --- [봇 모드] ---
            if self.is_botting and self.model:
                try:
                    ult_ready = 1 if self.skill_manager.is_ready("ultimate") else 0
                    sub_ready = 1 if self.skill_manager.is_ready("sub_attack") else 0
                    
                    # [핵심 수정] numpy array 대신 pandas DataFrame 사용 (경고 제거)
                    # Feature 순서: [px, py, entropy, pid, ult_ready, sub_ready]
                    # 주의: self.feature_cols가 로드되지 않았을 경우를 대비해 기본값 지정
                    cols = getattr(self, 'feature_cols', ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready'])
                    
                    input_df = pd.DataFrame([[px, py, entropy, pid, ult_ready, sub_ready]], columns=cols)
                    
                    # DataFrame을 넣으면 Scaler가 경고를 띄우지 않습니다.
                    feats_scaled = self.scaler.transform(input_df)
                    
                    self.history.append(feats_scaled[0])
                    
                    # 3. 추론 (데이터 10개 쌓이면)
                    if len(self.history) == 10:
                        inp = torch.FloatTensor(np.array([self.history])).to(self.device)
                        
                        with torch.no_grad():
                            out = self.model(inp)
                            _, pred = torch.max(out, 1)
                            action_name = self.encoder.inverse_transform([pred.item()])[0]
                        
                        active_skill = action_name
                        
                        # 4. 행동 실행
                        if action_name != "None":
                            keys = action_name.split('+')
                            for s_name, s_key in self.input_handler.key_map.items():
                                if s_key in keys: self.skill_manager.use(s_name)

                            for k in keys: self.input_handler.hold(k)
                            time.sleep(0.04) 
                            for k in keys: self.input_handler.release(k)
                            
                except Exception as e:
                    self.is_botting = False
                    print(f"Bot Error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("봇 런타임 오류", f"봇 실행 중 문제가 발생했습니다:\n{e}"))
                    self.root.after(0, lambda: self.btn_bot.config(text="🤖 AUTO HUNT (봇 가동)"))
                    self.root.after(0, lambda: self.lbl_bot_status.config(text="[BOT: ERROR]", foreground="red"))

            # GUI 업데이트
            self.root.after(0, self.update_gui, frame, entropy, active_skill, kill_count, px, py)
            time.sleep(0.033)

    # 기존 함수들 (가독성 복구 및 버그 수정)
    def open_map_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if file_path:
            if self.brain.load_map_file(file_path):
                self.lbl_map_name.config(text=file_path.split("/")[-1], foreground="green")
                messagebox.showinfo("성공", "맵 파일 로드 완료")

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

    def load_settings(self):
        data = utils.load_config()
        self.entry_job.insert(0, data.get("job_name", "Adventurer"))
        self.map_offset_x = data.get("map_offset_x", 0)
        self.map_offset_y = data.get("map_offset_y", 0)
        self.lbl_offset.config(text=f"Offset: ({self.map_offset_x}, {self.map_offset_y})")
        
        # 미니맵 ROI 복구
        minimap_roi = data.get("minimap_roi")
        if minimap_roi and isinstance(minimap_roi, (list, tuple)): # 값이 있고 리스트/튜플인지 확인
            self.vision.set_minimap_roi(tuple(minimap_roi))
            
        mapping = data.get("mapping", {})
        for r in self.skill_rows: r["frame"].destroy()
        self.skill_rows = []
        if not mapping: self.add_skill_row("Genesis", "r", "30.0")
        else:
            for s, i in mapping.items():
                self.add_skill_row(s, i.get("key", ""), str(i.get("cd", 0)))
        self.update_logic_from_ui()

    def save_settings(self):
        mapping = {}
        for r in self.skill_rows:
            # 삭제된 위젯에 접근하지 않도록 안전장치
            try:
                if r["frame"].winfo_exists() and r["name"].get():
                    mapping[r["name"].get()] = {"key": r["key"].get(), "cd": float(r["cd"].get() or 0)}
            except: pass
            
        data = {
            "job_name": self.entry_job.get(),
            "mapping": mapping,
            "map_offset_x": self.map_offset_x,
            "map_offset_y": self.map_offset_y,
            "minimap_roi": self.vision.minimap_roi
        }
        utils.save_config(data)
        self.update_logic_from_ui()
        messagebox.showinfo("저장됨", "설정이 저장되었습니다.")

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
        if self.is_recording:
            self.is_recording = False
            self.btn_record.config(text="⏺ REC (데이터 녹화)")
            if self.logger: self.logger.close(); messagebox.showinfo("완료", f"저장: {self.logger.filepath}")
            self.logger = None
        else:
            if not self.vision.window_found: messagebox.showwarning("경고", "창을 먼저 찾으세요."); return
            self.logger = DataLogger(self.entry_job.get())
            self.is_recording = True
            self.btn_record.config(text="⏹ STOP (저장 중...)", state="normal")

    def update_gui(self, frame, entropy, action, kill, px, py):
        if frame is not None:
            # 발판 그리기
            if self.brain.footholds:
                for (x1,y1,x2,y2) in self.brain.footholds:
                    cv2.line(frame, (x1+self.map_offset_x, y1+self.map_offset_y), 
                             (x2+self.map_offset_x, y2+self.map_offset_y), (0,0,255), 2)
            # 캐릭터 위치
            if self.vision.minimap_roi and px>0:
                mx, my, _, _ = self.vision.minimap_roi
                cv2.circle(frame, (mx+px, my+py), 5, (0,255,0), -1)
                
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(frame, (640,360)), cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=img, anchor="nw")
            self.canvas.image = img
            
        self.lbl_entropy.config(text=f"Ent: {entropy:.0f} | Pos: ({px},{py})")
        self.lbl_action.config(text=f"Act: {action}")
        self.lbl_kill.config(text=f"Kills: {kill}")
        
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