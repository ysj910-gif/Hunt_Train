# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from modules.vision import VisionSystem
from modules.brain import SkillManager, StrategyBrain
from modules.input import InputHandler
import utils

class MapleHunterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maple Hunter AI (Modular Ver.)")
        self.root.geometry("1100x750")

        # === 모듈 초기화 ===
        self.vision = VisionSystem()
        self.skill_manager = SkillManager()
        self.brain = StrategyBrain(self.skill_manager)
        self.input_handler = InputHandler()
        
        self.is_running = False
        self.ui_entries = {} # UI 입력창 저장소

        self.setup_ui()
        self.load_settings()
        self.map_path = None

    def setup_ui(self):
        # 레이아웃 생성
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)
        
        left = ttk.Frame(paned, padding=10)
        right = ttk.Frame(paned, padding=10)
        paned.add(left, weight=2)
        paned.add(right, weight=1)

        # [Left] 화면 & 상태
        self.canvas = tk.Canvas(left, bg="black", height=360)
        self.canvas.pack(fill="x", pady=5)
        
        status_frame = ttk.Frame(left)
        status_frame.pack(fill="x", pady=10)
        self.lbl_entropy = ttk.Label(status_frame, text="Entropy: 0", font=("Consolas", 14), foreground="blue")
        self.lbl_entropy.pack(side="left")
        self.lbl_action = ttk.Label(status_frame, text="Action: IDLE", font=("Consolas", 14, "bold"), foreground="red")
        self.lbl_action.pack(side="right")

        # 쿨타임 바
        self.pb_ultimate = self.create_pb(left, "Ultimate")
        self.pb_sub = self.create_pb(left, "Sub Attack")

        # [Right] 설정
        ttk.Label(right, text="Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 키 매핑 입력 생성
        for skill in self.skill_manager.cooldowns.keys():
            f = ttk.Frame(right)
            f.pack(fill="x", pady=2)
            ttk.Label(f, text=skill, width=10).pack(side="left")
            e_key = ttk.Entry(f, width=5); e_key.pack(side="left")
            e_cd = ttk.Entry(f, width=5); e_cd.pack(side="right")
            self.ui_entries[skill] = (e_key, e_cd)

        # 민감도
        ttk.Label(right, text="Entropy Threshold").pack(pady=(20, 0))
        self.scale_thresh = ttk.Scale(right, from_=500, to=10000, orient="horizontal")
        self.scale_thresh.pack(fill="x")
        
        # 버튼
        ttk.Button(right, text="Save Config", command=self.save_settings).pack(fill="x", pady=10)
        self.btn_start = ttk.Button(right, text="▶ START", command=self.toggle_running)
        self.btn_start.pack(fill="x", ipady=10)

        # gui.py 내 setup_ui 함수 안쪽 적절한 곳 (예: Start 버튼 위)

        self.btn_find_win = ttk.Button(right, text="🔍 메이플 창 자동 찾기", command=self.find_window_action)
        self.btn_find_win.pack(fill="x", pady=5)

        # === [추가된 부분] 맵 파일 로드 버튼 ===
        map_frame = ttk.LabelFrame(right, text="Map Data")
        map_frame.pack(fill="x", pady=5)
        
        self.lbl_map_name = ttk.Label(map_frame, text="No Map Loaded", foreground="gray")
        self.lbl_map_name.pack(side="left", padx=5)
        
        ttk.Button(map_frame, text="📂 Load JSON", command=self.open_map_file).pack(side="right", padx=5)
        # ========================================

    def create_pb(self, parent, text):
        f = ttk.Frame(parent); f.pack(fill="x", pady=2)
        ttk.Label(f, text=text, width=10).pack(side="left")
        pb = ttk.Progressbar(f, length=200); pb.pack(side="right", expand=True, fill="x")
        return pb

    def load_settings(self):
        data = utils.load_config()
        self.scale_thresh.set(data.get("threshold", 3000))
        mapping = data.get("mapping", {})
        
        for skill, (e_key, e_cd) in self.ui_entries.items():
            vals = mapping.get(skill, {"key": "", "cd": "0"})
            e_key.delete(0, tk.END); e_key.insert(0, vals["key"])
            e_cd.delete(0, tk.END); e_cd.insert(0, vals["cd"])
        last_map = data.get("last_map")
        if last_map and os.path.exists(last_map):
            self.map_path = last_map
            self.lbl_map_name.config(text=last_map.split("/")[-1], foreground="green")
            self.brain.load_map_file(last_map)

    def save_settings(self):
        data = {
            "threshold": self.scale_thresh.get(), 
            "mapping": {},
            "last_map": self.map_path # 맵 경로 저장
        }
        for skill, (e_key, e_cd) in self.ui_entries.items():
            data["mapping"][skill] = {"key": e_key.get(), "cd": e_cd.get()}
        utils.save_config(data)
        messagebox.showinfo("Saved", "설정이 저장되었습니다.")

    def open_map_file(self):
        """파일 선택 창 열기"""
        file_path = filedialog.askopenfilename(
            title="Select Map JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.map_path = file_path
            # 1. GUI 라벨 업데이트
            file_name = file_path.split("/")[-1]
            self.lbl_map_name.config(text=file_name, foreground="green")
            
            # 2. Brain에 맵 데이터 주입
            if self.brain.load_map_file(file_path):
                messagebox.showinfo("Success", f"맵 데이터 로드 완료!\n{file_name}")
            else:
                messagebox.showerror("Error", "맵 파일을 읽지 못했습니다.")

    def apply_ui_to_logic(self):
        # UI 값을 로직 모듈로 전송
        self.brain.threshold = self.scale_thresh.get()
        new_key_map = {}
        for skill, (e_key, e_cd) in self.ui_entries.items():
            try:
                self.skill_manager.set_cooldown(skill, float(e_cd.get()))
                new_key_map[skill] = e_key.get()
            except: pass
        self.input_handler.update_key_map(new_key_map)

    def toggle_running(self):
        if self.is_running:
            self.is_running = False
            self.btn_start.config(text="▶ START")
        else:
            self.apply_ui_to_logic()
            self.is_running = True
            self.btn_start.config(text="⏹ STOP")
            threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.is_running:
            # 1. Vision: 보고
            frame, entropy = self.vision.capture_and_analyze()
            
            # 2. Brain: 생각하고
            action = self.brain.decide_action(entropy)
            
            # 3. Hand: 누른다
            log = self.input_handler.execute(action)
            if action in ["ultimate", "sub_attack", "buff"]:
                self.skill_manager.use(action)

            # 4. GUI 업데이트
            self.root.after(0, self.update_gui, frame, entropy, action, log)
            
            time.sleep(random.uniform(0.1, 0.2)) # 루프 속도 조절

    def update_gui(self, frame, entropy, action, log):
        # 이미지
        frame_s = cv2.resize(frame, (320, 180))
        img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=img, anchor="nw")
        self.canvas.image = img
        
        # 텍스트
        self.lbl_entropy.config(text=f"Entropy: {entropy:.0f}")
        self.lbl_action.config(text=f"Action: {action} ({log})")
        
        # 쿨타임 바 업데이트 (수정된 부분)
        for skill, pb in [("ultimate", self.pb_ultimate), ("sub_attack", self.pb_sub)]:
            rem = self.skill_manager.get_remaining(skill)
            tot = self.skill_manager.cooldowns.get(skill, 0) # 기본값을 0으로
            
            # [수정] 쿨타임 총합이 0보다 클 때만 계산 (0으로 나누기 방지)
            if tot > 0:
                pb['value'] = ((tot - rem) / tot) * 100
            else:
                pb['value'] = 100 # 쿨타임이 0이면 항상 꽉 찬 상태

    # gui.py 의 MapleHunterUI 클래스 안쪽에 이 함수를 추가하세요.
    
    def find_window_action(self):
        # Vision 모듈에게 창 찾기 명령을 내림
        if self.vision.find_maple_window():
            messagebox.showinfo("성공", "메이플스토리 창을 찾아 좌표를 고정했습니다.")
        else:
            messagebox.showerror("실패", "메이플스토리 창을 찾을 수 없습니다.\n1. 게임 실행 여부\n2. 관리자 권한 실행 여부\n3. 창 모드 여부\n확인해주세요.")