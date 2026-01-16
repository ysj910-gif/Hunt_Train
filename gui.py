# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Toplevel
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
from pynput import keyboard

from modules.vision import VisionSystem
from modules.brain import SkillManager
from modules.input import InputHandler
from modules.logger import DataLogger
import utils
import config

class MapleHunterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maple Data Recorder (Custom Skill Ver.)")
        self.root.geometry("1200x850")

        self.vision = VisionSystem()
        self.skill_manager = SkillManager()
        self.logger = None 
        self.input_handler = InputHandler()
        
        self.is_recording = False
        self.current_key = "None"
        
        # 동적 스킬 행들을 저장할 리스트
        # 예: [{"frame": Frame, "name": Entry, "key": Entry, "cd": Entry}, ...]
        self.skill_rows = []
        self.key_to_skill_map = {} 

        self.setup_ui()
        self.load_settings()
        
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

        threading.Thread(target=self.loop, daemon=True).start()

        # [신규] 맵 오프셋 (픽셀 단위 조정)
        self.map_offset_x = 0
        self.map_offset_y = 0

        self.setup_ui()
        self.load_settings()
        
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

        threading.Thread(target=self.loop, daemon=True).start()

    def on_key_press(self, key):
        if self.is_recording:
            try:
                if hasattr(key, 'char') and key.char:
                    self.current_key = key.char
                else:
                    self.current_key = str(key).replace("Key.", "")
            except:
                self.current_key = "Unknown"

    def setup_ui(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)
        
        left = ttk.Frame(paned, padding=10)
        right = ttk.Frame(paned, padding=10)
        paned.add(left, weight=2)
        paned.add(right, weight=1)

        # === [Left] 화면 ===
        self.canvas = tk.Canvas(left, bg="black", height=360)
        self.canvas.pack(fill="x", pady=5)
        
        status_frame = ttk.Frame(left)
        status_frame.pack(fill="x", pady=10)
        self.lbl_entropy = ttk.Label(status_frame, text="Entropy: 0", font=("Consolas", 14), foreground="blue")
        self.lbl_entropy.pack(side="left", padx=5)
        self.lbl_kill = ttk.Label(status_frame, text="Kills: 0", font=("Consolas", 14, "bold"), foreground="green")
        self.lbl_kill.pack(side="left", padx=20)
        self.lbl_action = ttk.Label(status_frame, text="Action: None", font=("Consolas", 14, "bold"), foreground="red")
        self.lbl_action.pack(side="right", padx=5)

        # 쿨타임 표시 영역 (동적으로 생성됨)
        self.cooldown_frame = ttk.Frame(left)
        self.cooldown_frame.pack(fill="x", pady=5)

        # === [Right] 설정 ===
        
        # 1. 직업 정보
        job_frame = ttk.LabelFrame(right, text="Player Info")
        job_frame.pack(fill="x", pady=5)
        ttk.Label(job_frame, text="Job Class:").pack(side="left", padx=5)
        self.entry_job = ttk.Entry(job_frame)
        self.entry_job.pack(side="left", fill="x", expand=True, padx=5)

        # 2. 스킬 설정 (동적 리스트)
        setting_frame = ttk.LabelFrame(right, text="Custom Skills")
        setting_frame.pack(fill="both", expand=True, pady=5)

        # 스크롤 가능한 캔버스 영역 만들기 (스킬이 많아질 경우 대비)
        canvas_scroll = tk.Canvas(setting_frame, height=300)
        scrollbar = ttk.Scrollbar(setting_frame, orient="vertical", command=canvas_scroll.yview)
        self.skill_list_frame = ttk.Frame(canvas_scroll)

        self.skill_list_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        canvas_scroll.create_window((0, 0), window=self.skill_list_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 헤더
        h_frame = ttk.Frame(self.skill_list_frame)
        h_frame.pack(fill="x", pady=2)
        ttk.Label(h_frame, text="Skill Name", width=15, font="bold").pack(side="left", padx=2)
        ttk.Label(h_frame, text="Key", width=6, font="bold").pack(side="left", padx=2)
        ttk.Label(h_frame, text="CD(s)", width=6, font="bold").pack(side="left", padx=2)

# --- [Tab 2: Map & Offset] ---
        
        # 1. 맵 로드 버튼
        map_load_frame = ttk.LabelFrame(tab_map, text="Map File (JSON)")
        map_load_frame.pack(fill="x", pady=10, padx=5)
        
        self.lbl_map_name = ttk.Label(map_load_frame, text="No Map Loaded", foreground="gray")
        self.lbl_map_name.pack(pady=5)
        ttk.Button(map_load_frame, text="📂 Load Map JSON", command=self.open_map_file).pack(fill="x", padx=5, pady=5)
        
        # 2. 오프셋 조절 (미세조정)
        offset_frame = ttk.LabelFrame(tab_map, text="Foothold Position Fine-tuning")
        offset_frame.pack(fill="x", pady=10, padx=5)
        
        self.lbl_offset = ttk.Label(offset_frame, text="Offset: (X=0, Y=0)", font=("Arial", 10, "bold"))
        self.lbl_offset.pack(pady=5)
        
        # 화살표 버튼 배치
        btn_pad = ttk.Frame(offset_frame)
        btn_pad.pack(pady=5)
        
        # Grid를 이용해 화살표 모양으로 배치
        ttk.Button(btn_pad, text="▲", width=5, command=lambda: self.adjust_offset(0, -1)).grid(row=0, column=1, pady=2)
        ttk.Button(btn_pad, text="◀", width=5, command=lambda: self.adjust_offset(-1, 0)).grid(row=1, column=0, padx=2)
        ttk.Button(btn_pad, text="▼", width=5, command=lambda: self.adjust_offset(0, 1)).grid(row=1, column=1, pady=2)
        ttk.Button(btn_pad, text="▶", width=5, command=lambda: self.adjust_offset(1, 0)).grid(row=1, column=2, padx=2)
        
        # 리셋 버튼
        ttk.Button(offset_frame, text="Reset Offset", command=lambda: self.adjust_offset(0, 0, reset=True)).pack(pady=10)
        
        ttk.Label(offset_frame, text="* JSON 발판 좌표를 화면에 맞게 이동시킵니다.", foreground="gray").pack()       
# 3. 제어 버튼 영역 (control_frame 안쪽)
        control_frame = ttk.Frame(right)
        control_frame.pack(fill="x", pady=5)
        
        ttk.Button(control_frame, text="+ Add Skill", command=self.add_skill_row).pack(fill="x", pady=2)
        ttk.Button(control_frame, text="💾 Save Config & Update", command=self.save_settings).pack(fill="x", pady=5)

        self.btn_find_win = ttk.Button(right, text="1. 🔍 메이플 창 찾기", command=self.find_window_action)
        self.btn_find_win.pack(fill="x", pady=(10, 5))
        
        # [추가된 버튼] 킬 카운트 영역 지정
        self.btn_set_roi = ttk.Button(right, text="🎯 킬 카운트 영역 지정 (드래그)", command=self.open_roi_selector)
        self.btn_set_roi.pack(fill="x", pady=5)
        
        self.btn_record = ttk.Button(right, text="2. ⏺ REC (데이터 녹화 시작)", command=self.toggle_recording)
        self.btn_record.pack(fill="x", ipady=10, pady=5)

    def open_map_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Map JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            filename = file_path.split("/")[-1]
            if self.brain.load_map_file(file_path):
                self.lbl_map_name.config(text=filename, foreground="green")
                messagebox.showinfo("Load Success", f"{len(self.brain.footholds)}개의 발판 데이터를 불러왔습니다.\nOffset 탭에서 위치를 맞춰주세요.")
            else:
                self.lbl_map_name.config(text="Load Failed", foreground="red")
                messagebox.showerror("Error", "파일을 읽을 수 없거나 형식이 잘못되었습니다.")

    def adjust_offset(self, dx, dy, reset=False):
        if reset:
            self.map_offset_x = 0
            self.map_offset_y = 0
        else:
            self.map_offset_x += dx
            self.map_offset_y += dy
        
        self.lbl_offset.config(text=f"Offset: (X={self.map_offset_x}, Y={self.map_offset_y})")

    # --- [기존 로직 및 스킬 관련 메서드] ---
        
    def add_skill_row(self, name="", key="", cd="0.0"):
        """스킬 입력 줄 하나를 추가합니다."""
        row_f = ttk.Frame(self.skill_list_frame)
        row_f.pack(fill="x", pady=2)

        e_name = ttk.Entry(row_f, width=15)
        e_name.pack(side="left", padx=2)
        e_name.insert(0, name)

        e_key = ttk.Entry(row_f, width=6)
        e_key.pack(side="left", padx=2)
        e_key.insert(0, key)

        e_cd = ttk.Entry(row_f, width=6)
        e_cd.pack(side="left", padx=2)
        e_cd.insert(0, cd)

        # 삭제 버튼
        btn_del = ttk.Button(row_f, text="X", width=3, command=lambda: self.delete_skill_row(row_f))
        btn_del.pack(side="left", padx=5)

        self.skill_rows.append({
            "frame": row_f,
            "name": e_name,
            "key": e_key,
            "cd": e_cd
        })

    def delete_skill_row(self, row_frame):
        """해당 스킬 줄을 삭제합니다."""
        row_frame.destroy()
        # 리스트에서도 제거
        self.skill_rows = [r for r in self.skill_rows if r["frame"] != row_frame]

    def load_settings(self):
        data = utils.load_config()
        self.entry_job.insert(0, data.get("job_name", "Adventurer"))
        
        # 기존 스킬 행들 모두 삭제 (초기화)
        for r in self.skill_rows:
            r["frame"].destroy()
        self.skill_rows = []

        mapping = data.get("mapping", {})
        
        # 저장된 스킬이 없으면 기본값(예시) 몇 개 추가
        if not mapping:
            self.add_skill_row("Genesis", "r", "30.0")
            self.add_skill_row("Heal", "d", "0.0")
        else:
            for skill_name, info in mapping.items():
                self.add_skill_row(skill_name, info.get("key", ""), str(info.get("cd", 0)))
        
        self.update_logic_from_ui()

    def save_settings(self):
        mapping = {}
        for r in self.skill_rows:
            s_name = r["name"].get().strip()
            s_key = r["key"].get().strip()
            s_cd = r["cd"].get().strip()
            
            if s_name: # 이름이 비어있지 않으면 저장
                mapping[s_name] = {"key": s_key, "cd": float(s_cd) if s_cd else 0.0}

        data = {
            "job_name": self.entry_job.get(),
            "threshold": 3000,
            "mapping": mapping
        }
        utils.save_config(data)
        self.update_logic_from_ui()
        messagebox.showinfo("Saved", "설정이 저장되고 스킬 리스트가 업데이트되었습니다.")

    def update_logic_from_ui(self):
        """UI에 입력된 내용을 실제 로직(Brain, Input, Map)에 반영"""
        self.key_to_skill_map.clear()
        new_cooldowns = {}
        new_key_map = {}

        # 1. 스킬 매핑 및 쿨타임 정보 추출
        for r in self.skill_rows:
            s_name = r["name"].get().strip()
            s_key = r["key"].get().strip().lower()
            s_cd = r["cd"].get().strip()

            if s_name:
                cd_val = float(s_cd) if s_cd else 0.0
                new_cooldowns[s_name] = cd_val
                if s_key:
                    self.key_to_skill_map[s_key] = s_name
                    new_key_map[s_name] = s_key

        # 2. SkillManager 업데이트
        self.skill_manager.update_skill_list(new_cooldowns)
        
        # 3. InputHandler 업데이트
        self.input_handler.update_key_map(new_key_map)
        
        # 4. 왼쪽 화면의 쿨타임 바 다시 그리기
        for widget in self.cooldown_frame.winfo_children():
            widget.destroy()
            
        self.progress_bars = {}
        for skill_name in new_cooldowns:
            if new_cooldowns[skill_name] > 0: # 쿨타임이 있는 스킬만 표시
                f = ttk.Frame(self.cooldown_frame)
                f.pack(fill="x", pady=1)
                ttk.Label(f, text=skill_name, width=10, anchor="w").pack(side="left")
                pb = ttk.Progressbar(f, length=150)
                pb.pack(side="right", fill="x", expand=True)
                self.progress_bars[skill_name] = pb

        print(f"매핑 업데이트 완료: {len(new_cooldowns)}개 스킬")

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.btn_record.config(text="2. ⏺ REC (데이터 녹화 시작)")
            if self.logger:
                messagebox.showinfo("완료", f"저장 완료!\n{self.logger.filepath}")
            self.logger = None
        else:
            if not self.vision.window_found:
                messagebox.showwarning("경고", "먼저 메이플 창을 찾아주세요.")
                return
            
            job_name = self.entry_job.get()
            self.logger = DataLogger(job_name)
            self.is_recording = True
            self.btn_record.config(text="⏹ STOP (저장 중...)", state="normal")

    def loop(self):
        while True:
            if self.vision.window_found:
                frame, entropy, kill_count = self.vision.capture_and_analyze()
            else:
                frame, entropy, kill_count = None, 0, 0
                time.sleep(0.5)
                continue

            # 키 -> 스킬 변환
            skill_name = "Idle"
            if self.current_key != "None":
                skill_name = self.key_to_skill_map.get(self.current_key, f"Key:{self.current_key}")
                
                # [중요] 사용자가 키를 눌렀을 때 쿨타임 매니저에게 '사용했다'고 알림
                # (그래야 화면에 쿨타임 바가 움직임)
                if skill_name in self.skill_manager.cooldowns:
                    self.skill_manager.use(skill_name)

            if self.is_recording and self.logger:
                # 로그 저장
                self.logger.log_step(entropy, self.skill_manager, skill_name, self.current_key, kill_count)

            self.root.after(0, self.update_gui, frame, entropy, skill_name, kill_count)
            
            # 키 입력 초기화 (한 번 감지 후 리셋)
            if self.current_key != "None":
                 self.current_key = "None"
                 
            time.sleep(0.1)

    def update_gui(self, frame, entropy, skill_name, kill_count):
        if frame is not None and frame.shape[0] > 0:
            # [발판 시각화] Brain에 로드된 발판이 있다면, 오프셋을 적용해 그립니다.
            if self.brain.footholds:
                for fh in self.brain.footholds:
                    # JSON: (x1, y1, x2, y2)
                    # 화면 표시: x + offset_x, y + offset_y
                    x1 = int(fh[0] + self.map_offset_x)
                    y1 = int(fh[1] + self.map_offset_y)
                    x2 = int(fh[2] + self.map_offset_x)
                    y2 = int(fh[3] + self.map_offset_y)
                    
                    # 빨간색 선 (두께 2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            frame_s = cv2.resize(frame, (640, 360))
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=img, anchor="nw")
            self.canvas.image = img
        
        self.lbl_entropy.config(text=f"Entropy: {entropy:.0f}")
        self.lbl_action.config(text=f"Action: {skill_name}")
        self.lbl_kill.config(text=f"Kills: {kill_count}")

        if hasattr(self, 'progress_bars'):
            for s_name, pb in self.progress_bars.items():
                rem = self.skill_manager.get_remaining(s_name)
                tot = self.skill_manager.cooldowns.get(s_name, 1)
                if tot > 0: pb['value'] = ((tot - rem) / tot) * 100
                else: pb['value'] = 100

    def find_window_action(self):
        if self.vision.find_maple_window():
            messagebox.showinfo("성공", "창을 찾았습니다.\n오른쪽 설정에서 스킬을 추가하고 'Save Config'를 눌러주세요!")
        else:
            messagebox.showerror("실패", "메이플 창을 찾을 수 없습니다.")

    def open_roi_selector(self):
        if not self.vision.window_found:
            messagebox.showwarning("경고", "먼저 '메이플 창 찾기'를 수행해주세요.")
            return

        # 현재 화면 한 장 캡처
        frame, _, _ = self.vision.capture_and_analyze()
        if frame is None: return

        # 새 창(Toplevel) 열기
        self.roi_win = Toplevel(self.root)
        self.roi_win.title("숫자 부분만 드래그하세요")
        self.roi_win.attributes('-topmost', True) # 맨 위에 표시

        # 이미지를 Tkinter용으로 변환
        self.roi_cv_img = frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.roi_pil_img = Image.fromarray(img_rgb)
        self.roi_tk_img = ImageTk.PhotoImage(self.roi_pil_img)

        # 캔버스에 이미지 표시
        w, h = self.roi_pil_img.size
        self.roi_canvas = tk.Canvas(self.roi_win, width=w, height=h, cursor="cross")
        self.roi_canvas.pack()
        self.roi_canvas.create_image(0, 0, image=self.roi_tk_img, anchor="nw")

        # 마우스 이벤트 연결
        self.roi_canvas.bind("<ButtonPress-1>", self.on_roi_press)
        self.roi_canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self.on_roi_release)

        self.roi_start = None
        self.roi_rect = None

    def on_roi_press(self, event):
        self.roi_start = (event.x, event.y)
        # 기존 사각형 삭제
        if self.roi_rect:
            self.roi_canvas.delete(self.roi_rect)

    def on_roi_drag(self, event):
        if self.roi_start:
            x0, y0 = self.roi_start
            x1, y1 = event.x, event.y
            # 드래그 중인 사각형 그리기 (빨간색)
            if self.roi_rect:
                self.roi_canvas.delete(self.roi_rect)
            self.roi_rect = self.roi_canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    def on_roi_release(self, event):
        if self.roi_start:
            x0, y0 = self.roi_start
            x1, y1 = event.x, event.y
            
            # 좌표 정렬 (왼쪽위, 오른쪽아래)
            x_start, x_end = sorted([x0, x1])
            y_start, y_end = sorted([y0, y1])
            
            w = x_end - x_start
            h = y_end - y_start
            
            if w > 5 and h > 5: # 너무 작은 영역 무시
                # Vision 모듈에 ROI 전달
                self.vision.set_roi((x_start, y_start, w, h))
                messagebox.showinfo("설정 완료", f"영역이 설정되었습니다.\n좌표: {x_start}, {y_start}, {w}x{h}")
                self.roi_win.destroy() # 창 닫기