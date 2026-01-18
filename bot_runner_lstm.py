# bot_runner_lstm.py
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import threading
import sys
import os
import tkinter as tk
from tkinter import filedialog
from collections import deque

# 사용자 모듈 임포트
from modules.vision import VisionSystem
from modules.input import InputHandler
from modules.brain import StrategyBrain, SkillManager
import config

# === [AI 모델 클래스 정의 (수정됨)] ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        # 동적 설정을 위해 변수 저장
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # [수정] 하드코딩된 값(2, 128) 대신 저장된 설정값 사용
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class BotRunnerLSTM:
    def __init__(self):
        # UI 숨김
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.model_path = ""
        self.map_file = ""
        self.config_file = ""
        
        # 1. 파일 선택
        self.select_files()

        # 2. 모듈 초기화
        # [중요] 수정된 modules/vision.py (MSS 방식)가 필요함
        self.vision = VisionSystem() 
        self.input_handler = InputHandler()
        self.skill_manager = SkillManager()
        self.brain = StrategyBrain(self.skill_manager)
        
        # 3. 설정 및 모델 로드
        self.load_config()
        self.load_map()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 구동 장치: {self.device}")
        self.load_model()
        
        # 4. 상태 관리
        self.is_running = False
        self.history = deque(maxlen=10) # 시퀀스 길이 (학습 때와 맞춰야 함)
        self.pressed_keys = set() # 눌린 키 상태 관리

    def select_files(self):
        print("\n📂 [1/3] 학습된 모델 파일(.pth)을 선택하세요...")
        self.model_path = filedialog.askopenfilename(
            title="1. LSTM 모델 선택 (.pth)",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if not self.model_path: sys.exit("❌ 모델 미선택 종료")

        print("📂 [2/3] 맵 데이터 파일(.json)을 선택하세요...")
        self.map_file = filedialog.askopenfilename(
            title="2. 맵 데이터 선택 (.json)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        print("📂 [3/3] 봇 설정 파일(hunter_config.json)을 선택하세요...")
        self.config_file = filedialog.askopenfilename(
            title="3. 설정 파일 선택 (hunter_config.json)",
            filetypes=[("JSON Files", "*.json")],
            initialfile="hunter_config.json"
        )
        if not self.config_file: sys.exit("❌ 설정 미선택 종료")

    def load_config(self):
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.offset_x = data.get("map_offset_x", 0)
                self.offset_y = data.get("map_offset_y", 0)
                
                skill_map = {k: v.get("cd", 0) for k, v in data.get("mapping", {}).items()}
                dur_map = {k: 0 for k in data.get("mapping", {}).keys()} # 지속시간은 일단 0
                key_map = {k: v.get("key", "") for k, v in data.get("mapping", {}).items()}
                
                self.skill_manager.update_skill_list(skill_map, dur_map)
                self.input_handler.update_key_map(key_map)
                print(f"✅ 설정 로드: 오프셋({self.offset_x}, {self.offset_y})")
        except Exception as e:
            sys.exit(f"❌ 설정 로드 실패: {e}")

    def load_map(self):
        if self.map_file and os.path.exists(self.map_file):
            self.brain.load_map_file(self.map_file)
        else:
            print("⚠️ 맵 파일 없음: 발판 ID는 항상 -1입니다.")

    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.scaler = checkpoint['scaler']
            self.encoder = checkpoint['encoder']
            # 학습 시 사용된 시퀀스 길이 확인 (없으면 기본 10)
            seq_len = checkpoint.get('seq_length', 10)
            self.history = deque(maxlen=seq_len)
            
            # 모델 파라미터 복원
            input_size = checkpoint['input_size']
            hidden_size = checkpoint['hidden_size']
            num_layers = checkpoint['num_layers']
            num_classes = checkpoint['num_classes']
            
            self.model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            print(f"✅ 모델 로드 완료 (Seq: {seq_len}, Hidden: {hidden_size})")
            
        except Exception as e:
            sys.exit(f"❌ 모델 로드 오류: {e}")

    def find_platform_id(self, px, py):
        """현재 좌표와 가장 가까운 발판 ID 찾기"""
        if not self.brain.footholds: return -1
        
        # 맵 파일 좌표계에 맞춰 오프셋 적용
        # (주의: 맵 파일이 미니맵 좌표 기준이라면 offset 더하기/빼기 방향 확인 필요)
        # 보통 미니맵 픽셀좌표 = 실제좌표 + offset 방식이 많음
        
        best_id = -1
        min_dist = 9999
        
        for i, (x1, y1, x2, y2) in enumerate(self.brain.footholds):
            # 발판 x범위 내에 있는지 확인 (오차범위 5픽셀)
            if (x1 - 5) <= px <= (x2 + 5):
                dist = abs(py - y1) # y축 거리 (발판 높이)
                if dist < min_dist:
                    min_dist = dist
                    best_id = i
        
        # 거리가 너무 멀면(예: 30픽셀 이상) 허공으로 판정
        if min_dist > 30: 
            return -1
            
        return best_id

    def update_key_state(self, action_str):
        """키 입력 동기화 (누르고 떼기)"""
        if action_str == 'None':
            target_keys = set()
        else:
            target_keys = set(action_str.split('+'))

        # 1. 떼야 할 키
        for k in list(self.pressed_keys):
            if k not in target_keys:
                self.input_handler.release(k)
                self.pressed_keys.remove(k)
        
        # 2. 눌러야 할 키
        for k in target_keys:
            if k not in self.pressed_keys:
                self.input_handler.hold(k)
                self.pressed_keys.add(k)

    def run(self):
        print("\n👀 메이플스토리 창 찾는 중...")
        while not self.vision.find_maple_window():
            time.sleep(1)
            
        print("\n▶️ LSTM 봇 시작! (Ctrl+C로 중단)")
        self.is_running = True
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # 1. 비전 인식 (수정된 vision.py 사용 시 검은 화면 없음)
                # 반환값: 프레임, 엔트로피, 킬카운트, x, y
                frame, entropy, _, raw_px, raw_py = self.vision.capture_and_analyze()
                
                if frame is None or frame.size == 0:
                    time.sleep(0.5); continue

                # 2. 좌표 보정 및 정보 추출
                # 오프셋은 설정 파일에 따라 다를 수 있으므로 확인 필요
                px = raw_px - self.offset_x
                py = raw_py - self.offset_y
                
                pid = self.find_platform_id(px, py)
                ult = 1 if self.skill_manager.is_ready("ultimate") else 0
                sub = 1 if self.skill_manager.is_ready("sub_attack") else 0
                
                # 3. 데이터 준비 (6 features)
                # ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready']
                features = np.array([[px, py, entropy, pid, ult, sub]])
                
                try:
                    features_scaled = self.scaler.transform(features)
                    self.history.append(features_scaled[0])
                except Exception as e:
                    print(f"⚠️ 데이터 전처리 오류: {e}")
                    continue
                
                # 4. 추론 및 행동
                if len(self.history) == self.history.maxlen:
                    input_seq = np.array([self.history])
                    input_tensor = torch.FloatTensor(input_seq).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        action_name = self.encoder.inverse_transform([predicted.item()])[0]
                    
                    # 행동 실행
                    if action_name != "None":
                        # print(f"🤖 Act: {action_name} | Pos: {px},{py}")
                        self.update_key_state(action_name)
                    else:
                        self.update_key_state("None")
                
                # FPS 조절
                elapsed = time.time() - loop_start
                if elapsed < 0.033: time.sleep(0.033 - elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 중단됨.")
        except Exception as e:
            print(f"\n❌ 런타임 오류: {e}")
        finally:
            self.input_handler.release_all()
            print("봇 종료.")

if __name__ == "__main__":
    BotRunnerLSTM().run()