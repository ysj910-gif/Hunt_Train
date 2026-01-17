# bot_runner_lstm_v2.py
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib
import threading
import sys
import os
import tkinter as tk
from tkinter import filedialog
from collections import deque

# 사용자 모듈 임포트 (파일이 같은 폴더에 있어야 함)
from modules.vision import VisionSystem
from modules.input import InputHandler
from modules.brain import StrategyBrain, SkillManager
import config

# === [LSTM 모델 클래스 (학습 코드와 구조가 같아야 함)] ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 학습된 가중치 구조에 맞춰 hidden state 초기화
        h0 = torch.zeros(2, x.size(0), 128).to(x.device) 
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class BotRunnerLSTM:
    def __init__(self):
        # 파일 선택을 위한 루트 윈도우 생성 (숨김)
        self.root = tk.Tk()
        self.root.withdraw()
        
        # 파일 경로 변수 초기화
        self.model_path = ""
        self.map_file = ""
        self.config_file = ""
        
        # 1. 파일 선택 진행
        self.select_files()

        # 2. 모듈 초기화
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
        self.history = deque(maxlen=10) # 10프레임 기억 저장소

    def select_files(self):
        """사용자에게 필요한 파일 3개를 순서대로 선택받음"""
        print("\n📂 [1/3] 학습된 모델 파일(.pth)을 선택하세요...")
        self.model_path = filedialog.askopenfilename(
            title="1. 학습된 모델 파일 선택 (.pth)",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if not self.model_path:
            print("❌ 모델 파일이 선택되지 않아 종료합니다."); sys.exit()

        print("📂 [2/3] 맵 데이터 파일(.json)을 선택하세요...")
        self.map_file = filedialog.askopenfilename(
            title="2. 맵 데이터 파일 선택 (.json)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not self.map_file:
            print("⚠️ 맵 파일이 선택되지 않았습니다. (발판 인식 기능 제한됨)")

        print("📂 [3/3] 봇 설정 파일(hunter_config.json)을 선택하세요...")
        self.config_file = filedialog.askopenfilename(
            title="3. 봇 설정 파일 선택 (hunter_config.json)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile="hunter_config.json"
        )
        if not self.config_file:
            print("❌ 설정 파일이 선택되지 않아 종료합니다."); sys.exit()
            
        print("\n✅ 모든 파일 선택 완료!")
        print(f" - Model: {os.path.basename(self.model_path)}")
        print(f" - Map: {os.path.basename(self.map_file) if self.map_file else 'None'}")
        print(f" - Config: {os.path.basename(self.config_file)}\n")

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.offset_x = data.get("map_offset_x", 0)
                self.offset_y = data.get("map_offset_y", 0)
                
                skill_map = {}
                dur_map = {}
                key_map = {}
                for name, info in data.get("mapping", {}).items():
                    skill_map[name] = info.get("cd", 0)
                    dur_map[name] = 0
                    key_map[name] = info.get("key", "")
                
                self.skill_manager.update_skill_list(skill_map, dur_map)
                self.input_handler.update_key_map(key_map)
                print(f"✅ 설정 로드 완료 (Offset: X={self.offset_x}, Y={self.offset_y})")
        except Exception as e:
            print(f"❌ 설정 로드 실패: {e}"); sys.exit()

    def load_map(self):
        if self.map_file and os.path.exists(self.map_file):
            self.brain.load_map_file(self.map_file)
        else:
            print("⚠️ 맵 파일이 로드되지 않았습니다. 발판 ID는 항상 -1이 됩니다.")

    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.scaler = checkpoint['scaler']
            self.encoder = checkpoint['encoder']
            # 저장된 feature_cols가 없으면 기본값 사용
            self.feature_cols = checkpoint.get('feature_cols', ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready'])
            
            input_size = checkpoint['input_size']
            hidden_size = checkpoint['hidden_size']
            num_layers = checkpoint['num_layers']
            num_classes = checkpoint['num_classes']
            
            self.model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            print(f"✅ 모델 로드 성공! (입력: {input_size}, 클래스: {num_classes})")
            
        except Exception as e:
            print(f"❌ 모델 로드 오류: {e}"); sys.exit()

    def find_platform_id(self, px, py):
        if not self.brain.footholds: return -1
        best_id = -1; min_dist = 50
        for i, (x1, y1, x2, y2) in enumerate(self.brain.footholds):
            fx1 = x1 + self.offset_x; fy = y1 + self.offset_y; fx2 = x2 + self.offset_x
            if fx1 <= px <= fx2:
                dist = abs(py - fy)
                if dist < min_dist: min_dist = dist; best_id = i
        return best_id

    def run(self):
        print("\n👀 메이플스토리 창을 찾는 중...")
        while not self.vision.find_maple_window():
            time.sleep(1)
            
        print("\n▶️ 봇 가동 시작! (중단하려면 터미널에서 Ctrl+C)")
        self.is_running = True
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # 1. 화면 인식
                frame, entropy, _, px, py = self.vision.capture_and_analyze()
                
                # 2. 정보 가공
                pid = self.find_platform_id(px, py)
                ult_ready = 1 if self.skill_manager.is_ready("ultimate") else 0
                sub_ready = 1 if self.skill_manager.is_ready("sub_attack") else 0
                
                # 3. 데이터 패키징
                features = np.array([[px, py, entropy, pid, ult_ready, sub_ready]])
                features_scaled = self.scaler.transform(features)
                
                # 4. 기억(History) 추가 및 추론
                self.history.append(features_scaled[0])
                
                if len(self.history) == 10:
                    input_seq = np.array([self.history])
                    input_tensor = torch.FloatTensor(input_seq).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        action_name = self.encoder.inverse_transform([predicted.item()])[0]
                    
                    # 5. 행동 실행
                    if action_name != "None":
                        print(f"🤖 Act: {action_name:<15} | Pos: ({px},{py})")
                        keys = action_name.split('+')
                        
                        # 쿨타임 갱신
                        for s_name, s_key in self.input_handler.key_map.items():
                            if s_key in keys: self.skill_manager.use(s_name)

                        # 키 입력 (동시 입력 처리)
                        for k in keys: self.input_handler.hold(k)
                        time.sleep(0.04) # 짧게 누름
                        for k in keys: self.input_handler.release(k)
                
                # FPS 유지
                elapsed = time.time() - loop_start
                if elapsed < 0.033: time.sleep(0.033 - elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 사용자 중단.")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
        finally:
            self.input_handler.release_all()
            print("봇 종료.")

if __name__ == "__main__":
    BotRunnerLSTM().run()