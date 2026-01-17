import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import pandas as pd
import cv2
import numpy as np
from pynput import keyboard
import pygetwindow as gw # 창 위치 찾기용
import winsound
import mss
import ctypes
# Windows DPI 인식 설정 (좌표 정밀도 확보)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    ctypes.windll.user32.SetProcessDPIAware()

import config
from vision_engine import VisionEngine
from platform_manager import PlatformManager

from ctypes import wintypes

user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG)]

def get_client_area_on_screen(hwnd):
    """
    returns (client_left, client_top, client_width, client_height) in SCREEN coords
    """
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()

    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise ctypes.WinError()

    w = rect.right - rect.left
    h = rect.bottom - rect.top
    return pt.x, pt.y, w, h

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("메이플 데이터 수집기 (GUI 모드)")
        self.root.geometry("500x700")

        self.vision = VisionEngine()
        self.plat_mgr = PlatformManager()
        
        # 상태 변수
        self.is_recording = False
        self.is_preview = False  # <--- [추가] 미리보기 상태 변수
        self.samples = []
        self.current_actions = set()
        self.map_loaded = False
        self.start_time = 0
        self.window_rect = None # 메이플 창 좌표

        # GUI 변수들
        self.path_var = tk.StringVar(value=config.DATA_DIR)
        self.file_var = tk.StringVar(value=config.SAVE_NAME)
        self.job_var = tk.StringVar(value=config.JOB)
        self.move_type_var = tk.StringVar(value=config.MOVE_TYPE)
        self.jump_key_var = tk.StringVar(value="alt")
        self.special_key_var = tk.StringVar(value="space")
        self.target_time_var = tk.StringVar(value=str(config.TARGET_RECORD_TIME))

        # 좌표 보정 변수 (초기값 0)
        self.offset_x = tk.IntVar(value=0)
        self.offset_y = tk.IntVar(value=0)

        self._setup_layout()

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)



    def _setup_layout(self):
        main_scroll = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_scroll.yview)
        main_frame = ttk.Frame(main_scroll, padding="20")
        
        main_scroll.create_window((0, 0), window=main_frame, anchor="nw")
        main_scroll.configure(yscrollcommand=scrollbar.set)
        
        main_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def add_section(parent, title):
            frame = ttk.LabelFrame(parent, text=title, padding="10")
            frame.pack(fill="x", pady=5)
            return frame

        # 1. 저장 설정
        save_f = add_section(main_frame, "1. 저장 경로 및 파일명")
        ttk.Label(save_f, text="경로:").grid(row=0, column=0, sticky="w")
        ttk.Entry(save_f, textvariable=self.path_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(save_f, text="파일명:").grid(row=1, column=0, sticky="w")
        ttk.Entry(save_f, textvariable=self.file_var).grid(row=1, column=1, sticky="ew", padx=5)

        # 2. 직업 및 이동방식 (체크박스/라디오 형태)
        job_f = add_section(main_frame, "2. 캐릭터 설정")
        ttk.Label(job_f, text="직업:").pack(side="left")
        ttk.Entry(job_f, textvariable=self.job_var, width=15).pack(side="left", padx=5)
        
        ttk.Radiobutton(job_f, text="더블점프(DJ)", variable=self.move_type_var, value="DJ").pack(side="left")
        ttk.Radiobutton(job_f, text="텔레포트(TP)", variable=self.move_type_var, value="TP").pack(side="left")

        # 3. 조작키 설정
        key_f = add_section(main_frame, "3. 조작키 설정")
        ttk.Label(key_f, text="점프 키:").grid(row=0, column=0)
        ttk.Entry(key_f, textvariable=self.jump_key_var, width=10).grid(row=0, column=1)
        ttk.Label(key_f, text="특수 이동기:").grid(row=0, column=2)
        ttk.Entry(key_f, textvariable=self.special_key_var, width=10).grid(row=0, column=3)

        # 4. 미니맵 및 맵 데이터
        map_f = add_section(main_frame, "4. 맵 및 미니맵 영역")
        ttk.Button(map_f, text="미니맵 영역 지정 (MapleStory 실행 필수)", command=self.set_minimap_roi).pack(fill="x", pady=2)
        ttk.Button(map_f, text="JSON 맵 파일 불러오기", command=self.upload_map).pack(fill="x", pady=2)
        self.map_status = ttk.Label(map_f, text="맵 파일: 대기 중", foreground="red")
        self.map_status.pack()

        # 5. 녹화 제어 및 시간
        rec_f = add_section(main_frame, "5. 녹화 제어")
        ttk.Label(rec_f, text="목표 시간(초):").pack(side="left")
        ttk.Entry(rec_f, textvariable=self.target_time_var, width=10).pack(side="left", padx=5)
        
        self.time_label = ttk.Label(rec_f, text="현재 시간: 00:00", font=("Arial", 10, "bold"), foreground="blue")
        self.time_label.pack(pady=5)
        
        self.record_btn = ttk.Button(rec_f, text="녹화 시작", command=self.toggle_recording, state="disabled")
        self.record_btn.pack(fill="x", pady=5)
        self.count_label = ttk.Label(rec_f, text="수집된 프레임: 0")
        self.count_label.pack()

        main_frame.update_idletasks()
        main_scroll.config(scrollregion=main_scroll.bbox("all"))

        # 6. 좌표 미세 보정 (Manual Offset)
        adj_f = add_section(main_frame, "6. 좌표 미세 보정 (Offset)")
        ttk.Label(adj_f, text="X축 이동:").grid(row=0, column=0, padx=5)
        ttk.Spinbox(adj_f, from_=-100, to=100, textvariable=self.offset_x, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(adj_f, text="Y축 이동:").grid(row=0, column=2, padx=5)
        ttk.Spinbox(adj_f, from_=-100, to=100, textvariable=self.offset_y, width=10).grid(row=0, column=3, padx=5)

        ttk.Label(adj_f, text="(발판 선에 맞게 숫자를 조절하세요)", foreground="gray").grid(row=1, column=0, columnspan=4, pady=5)

        # [추가] 미리보기 버튼
        self.preview_btn = ttk.Button(adj_f, text="미리보기 시작 (저장 안 함)", command=self.toggle_preview)
        self.preview_btn.grid(row=2, column=0, columnspan=4, pady=5, sticky="ew")

    def find_maple_window(self):
    # 대소문자 구분 없이 'maplestory'가 포함된 모든 창 탐색
        all_windows = gw.getAllWindows()
        target = [w for w in all_windows if 'maplestory' in w.title.lower()]
        if not target:
            raise Exception("메이플스토리 창을 찾을 수 없습니다. (관리자 권한 실행 확인)")
        return target[0]

    def set_minimap_roi(self):
        """메이플 창 내부에서 미니맵 영역을 상대 좌표로 지정"""
        try:
            titles = [t for t in gw.getAllTitles() if 'MapleStory' in t]
            if not titles: raise Exception("메이플스토리 창을 찾을 수 없습니다.")
            win = gw.getWindowsWithTitle(titles[0])[0]
            win.activate()
            
            # 제목 표시줄 제외를 위해 창 좌표 획득 (대략적 수정 필요시 offset 조절)
            # win.left, win.top은 제목 표시줄을 포함한 창 전체의 좌상단입니다.
            with mss.mss() as sct:
                hwnd = win._hWnd
                c_left, c_top, c_w, c_h = get_client_area_on_screen(hwnd)

                monitor = {"top": c_top, "left": c_left, "width": c_w, "height": c_h}
                img = np.array(sct.grab(monitor))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                roi = cv2.selectROI("Select Minimap ROI", img_bgr, False)
                cv2.destroyWindow("Select Minimap ROI")

                if roi != (0, 0, 0, 0):
                # ROI는 이제 "클라이언트(게임 화면)" 기준 좌표가 됨
                    config.MINIMAP_ROI = {
                        'left': int(roi[0]),
                        'top': int(roi[1]),
                        'width': int(roi[2]),
                        'height': int(roi[3])
                    }
                    messagebox.showinfo("완료", "미니맵 영역 설정이 완료되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def upload_map(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.plat_mgr.load_platforms(file_path)
            self.map_loaded = True
            self.map_status.config(text=f"맵 파일: {os.path.basename(file_path)}", foreground="green")
            self.record_btn.config(state="normal")

    def apply_config(self):
        """GUI의 값들을 config에 반영"""
        config.DATA_DIR = self.path_var.get()
        config.SAVE_NAME = self.file_var.get()
        config.JOB = self.job_var.get()
        config.MOVE_TYPE = self.move_type_var.get()
        config.KEY_ACTION_MAP = {
            self.jump_key_var.get(): 'jump',
            self.special_key_var.get(): 'move_special',
            'left': 'move_left', 'right': 'move_right', 'up': 'move_up', 'down': 'move_down'
        }
        os.makedirs(os.path.join(config.DATA_DIR, "images"), exist_ok=True)

    def toggle_recording(self):
        if not self.is_recording:
            self.apply_config()
            self.is_recording = True
            self.start_time = time.time()
            self.alert_fired = False
            self.samples = []
            self.record_btn.config(text="녹화 중지 및 저장")
            threading.Thread(target=self.record_loop, daemon=True).start()
        else:
            self.stop_recording()

    def record_loop(self):
        count = 0
        interval = 1.0 / config.FPS_LIMIT
        target_sec = int(self.target_time_var.get())
        
        # 윈도우 캡처를 위한 mss 객체 생성
        with mss.mss() as sct:
            while self.is_recording:
                loop_start = time.time()
                elapsed = loop_start - self.start_time
                
                try:
                    # [수정됨] 1. 메이플 창 핸들 가져오기 및 클라이언트 영역 좌표 계산
                    windows = gw.getWindowsWithTitle('MapleStory')
                    if not windows:
                        print("메이플스토리 창을 찾을 수 없습니다.")
                        time.sleep(1)
                        continue
                        
                    win = windows[0]
                    
                    # 창 위치가 이동했을 수 있으므로 현재 위치를 다시 계산합니다.
                    # get_client_area_on_screen 함수는 파일 상단에 정의되어 있습니다.
                    c_left, c_top, c_w, c_h = get_client_area_on_screen(win._hWnd)
                    
                    # 캡처 영역 계산 (c_left, c_top 변수가 이제 정의됨)
                    capture_roi = {
                        "top": c_top + config.MINIMAP_ROI['top'],
                        "left": c_left + config.MINIMAP_ROI['left'],
                        "width": config.MINIMAP_ROI['width'],
                        "height": config.MINIMAP_ROI['height']
                    }
                    
                    # 2. 화면 캡처 및 이미지 변환
                    mini_img = np.array(sct.grab(capture_roi))
                    mini_bgr = cv2.cvtColor(mini_img, cv2.COLOR_BGRA2BGR)
                    
                    # ... (이하 코드는 기존과 동일) ...
                    # 3. 캐릭터 기본 탐지 (원본 좌표 raw_x, raw_y)
                    mask = self.vision.get_character_mask(mini_bgr)
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        raw_x = int(M["m10"] / M["m00"])
                        raw_y = int(M["m01"] / M["m00"])
                    else:
                        raw_x, raw_y = 0, 0

                    # 4. [보정치 가져오기]
                    # offset_x, y는 "발판을 얼마나 이동시켜서 그릴지"를 결정합니다.
                    off_x = self.offset_x.get()
                    off_y = self.offset_y.get()

                except Exception as e:
                    print(f"캡처/탐지 중 오류: {e}")
                    continue

                # 5. 발판 판정 및 거리 계산
                # 시각적으로는 발판을 (off_x, off_y)만큼 이동시켰으므로,
                # 논리적으로는 캐릭터 위치에서 offset을 빼야 원본 JSON 좌표계와 매칭됩니다.
                # 예: 화면상 발판이 Y=130에 있고(원본 100 + 오프셋 30), 캐릭터가 130에 있다면
                #     캐릭터(130) - 오프셋(30) = 100 -> JSON의 100과 일치함.
                calc_x = raw_x - off_x
                calc_y = raw_y - off_y
                
                current_plat = self.plat_mgr.get_current_platform(calc_x, calc_y)
                plat_id = current_plat.get('id', -1) if current_plat else -1
                
                # 6. 디버그 시각화 (이미지 위에 정보 덮어쓰기)
                debug_img = mini_bgr.copy()

                # (1) JSON 발판 그리기 (오프셋 적용하여 시각화)
                platforms = getattr(self.plat_mgr, 'platforms', [])
                for p in platforms:
                    # 발판 선을 offset만큼 이동시켜서 그립니다 (사용자가 맞추기 편하도록)
                    cv2.line(debug_img, 
                             (int(p['x_start'] + off_x), int(p['y'] + off_y)), 
                             (int(p['x_end'] + off_x), int(p['y'] + off_y)), 
                             (255, 255, 0), 1)

                # (2) 캐릭터 위치 표시
                if raw_x > 0:
                    # 인식된 실제 캐릭터 위치 (빨간색/초록색 점)
                    # 여기는 raw 좌표 그대로 찍어서 "실제 이미지상 위치"를 보여줍니다.
                    color = (0, 255, 0) if plat_id != -1 else (0, 0, 255)
                    cv2.circle(debug_img, (raw_x, raw_y), 4, color, -1)
                    
                    # 상단 정보 텍스트
                    info_text = f"ID: {plat_id} | RawPos: {raw_x},{raw_y} | Offset: {off_x},{off_y}"
                    cv2.putText(debug_img, info_text, (5, 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 디버그 창 출력
                cv2.imshow("Detection Debug", debug_img)
                cv2.waitKey(1)

                # 7. 데이터 저장
                img_name = f"{config.JOB}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(os.path.join(config.DATA_DIR, "images", img_name), mini_bgr)
                
                self.samples.append({
                    'timestamp': elapsed, 
                    'job': config.JOB, 
                    'image_path': img_name,
                    # 학습 데이터에는 JSON 기준 좌표(보정된 좌표)를 저장합니다.
                    'char_x': calc_x, 
                    'char_y': calc_y, 
                    'platform_id': plat_id,
                    'actions': " ".join(list(self.current_actions)) if self.current_actions else "idle"
                })
                
                # 8. UI 상태 업데이트 및 시간 초과 알람
                count += 1
                self.count_label.config(text=f"수집된 프레임: {count}")
                
                mins, secs = divmod(int(elapsed), 60)
                self.time_label.config(text=f"현재 시간: {mins:02d}:{secs:02d} / {target_sec}초")
                
                if elapsed >= target_sec and not self.alert_fired:
                    winsound.Beep(1000, 500)
                    self.alert_fired = True

                # FPS 제한을 위한 대기
                time.sleep(max(0, interval - (time.time() - loop_start)))
        
        # 녹화 종료 시 창 닫기
        cv2.destroyAllWindows()


    def stop_recording(self):
        self.is_recording = False
        self.record_btn.config(text="녹화 시작")
        if self.samples:
            df = pd.DataFrame(self.samples)
            csv_path = os.path.join(config.DATA_DIR, config.SAVE_NAME)
            df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
            messagebox.showinfo("저장 완료", f"{len(self.samples)} 프레임 저장됨")

    def toggle_preview(self):
        if not self.is_preview:
            # 녹화 중이면 미리보기 불가
            if self.is_recording:
                messagebox.showwarning("경고", "녹화 중에는 미리보기를 별도로 켤 수 없습니다.")
                return

            self.is_preview = True
            self.preview_btn.config(text="미리보기 중지")
            threading.Thread(target=self.preview_loop, daemon=True).start()
        else:
            self.is_preview = False
            self.preview_btn.config(text="미리보기 시작 (저장 안 함)")

    def preview_loop(self):
        """데이터 저장 없이 화면 인식 결과만 보여주는 루프"""
        with mss.mss() as sct:
            while self.is_preview:
                loop_start = time.time()
                
                try:
                    # 1. 메이플 창 찾기 및 위치 갱신 (오류 수정됨)
                    windows = gw.getWindowsWithTitle('MapleStory')
                    if not windows:
                        time.sleep(1)
                        continue
                    win = windows[0]
                    
                    # 창의 실제 클라이언트 영역(내용물) 좌표 계산
                    c_left, c_top, c_w, c_h = get_client_area_on_screen(win._hWnd)
                    
                    capture_roi = {
                        "top": c_top + config.MINIMAP_ROI['top'],
                        "left": c_left + config.MINIMAP_ROI['left'],
                        "width": config.MINIMAP_ROI['width'],
                        "height": config.MINIMAP_ROI['height']
                    }
                    
                    if capture_roi["width"] == 0 or capture_roi["height"] == 0:
                        print("미니맵 ROI가 설정되지 않았습니다.")
                        time.sleep(1)
                        continue

                    # 2. 캡처 및 인식
                    mini_img = np.array(sct.grab(capture_roi))
                    mini_bgr = cv2.cvtColor(mini_img, cv2.COLOR_BGRA2BGR)
                    
                    mask = self.vision.get_character_mask(mini_bgr)
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        raw_x = int(M["m10"] / M["m00"])
                        raw_y = int(M["m01"] / M["m00"])
                    else:
                        raw_x, raw_y = 0, 0

                    # 3. 오프셋 적용 및 발판 매칭 테스트
                    off_x = self.offset_x.get()
                    off_y = self.offset_y.get()
                    
                    calc_x = raw_x - off_x
                    calc_y = raw_y - off_y
                    
                    current_plat = self.plat_mgr.get_current_platform(calc_x, calc_y)
                    plat_id = current_plat.get('id', -1) if current_plat else -1

                    # 4. 디버그 그리기
                    debug_img = mini_bgr.copy()
                    
                    # (1) 발판 선 그리기 (오프셋 적용됨)
                    platforms = getattr(self.plat_mgr, 'platforms', [])
                    for p in platforms:
                        cv2.line(debug_img, 
                                 (int(p['x_start'] + off_x), int(p['y'] + off_y)), 
                                 (int(p['x_end'] + off_x), int(p['y'] + off_y)), 
                                 (255, 255, 0), 1)

                    # (2) 캐릭터 위치 및 정보 표시
                    if raw_x > 0:
                        color = (0, 255, 0) if plat_id != -1 else (0, 0, 255)
                        cv2.circle(debug_img, (raw_x, raw_y), 4, color, -1)
                        
                        info_text = f"ID: {plat_id} | Raw: {raw_x},{raw_y} | Off: {off_x},{off_y}"
                        cv2.putText(debug_img, info_text, (5, 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imshow("Preview Mode", debug_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.toggle_preview()
                        break

                except Exception as e:
                    print(f"미리보기 오류: {e}")
                
                # CPU 점유율 조절 (약 30 FPS)
                time.sleep(max(0, 0.033 - (time.time() - loop_start)))

        cv2.destroyWindow("Preview Mode")

    def on_press(self, key):
        try:
            k = key.char if hasattr(key, 'char') and key.char else str(key).replace('Key.', '')
            action = config.KEY_ACTION_MAP.get(k)
            if action: self.current_actions.add(action)
        except: pass

    def on_release(self, key):
        try:
            k = key.char if hasattr(key, 'char') and key.char else str(key).replace('Key.', '')
            action = config.KEY_ACTION_MAP.get(k)
            if action in self.current_actions: self.current_actions.remove(action)
        except: pass

    def on_closing(self):
        self.is_recording = False
        self.listener.stop()
        self.root.destroy()