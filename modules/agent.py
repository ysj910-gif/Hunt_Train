# modules/agent.py
import torch
import numpy as np
import pandas as pd
import joblib
import random
from collections import deque
from modules.model import LSTMModel

class BotAgent:
    def __init__(self):
        # 장치 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Agent Device: {self.device}")

        # 모델 및 상태 변수
        self.lstm_model = None
        self.rf_model = None
        self.scaler = None
        self.encoder = None
        
        # 기억 저장소
        self.seq_length = 10
        self.history = deque(maxlen=self.seq_length)
        
        # 입력 특성 컬럼 (기본값)
        self.feature_cols = ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready']

    def load_lstm(self, file_path):
        """LSTM 모델 로드"""
        try:
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            self.scaler = checkpoint['scaler']
            self.encoder = checkpoint['encoder']
            self.feature_cols = checkpoint.get('feature_cols', self.feature_cols)
            
            # 시퀀스 길이 업데이트
            self.seq_length = checkpoint.get('seq_length', 10)
            self.history = deque(maxlen=self.seq_length)
            
            # 모델 생성
            self.lstm_model = LSTMModel(
                checkpoint.get('input_size', 6),
                checkpoint.get('hidden_size', 128),
                checkpoint.get('num_layers', 2),
                checkpoint.get('num_classes', 10)
            ).to(self.device)
            
            self.lstm_model.load_state_dict(checkpoint['model_state'])
            self.lstm_model.eval()
            return True, f"LSTM Loaded (Seq: {self.seq_length})"
        except Exception as e:
            return False, str(e)

    def load_rf(self, file_path):
        """Random Forest 모델 로드"""
        try:
            self.rf_model = joblib.load(file_path)
            return True, "RF Loaded"
        except Exception as e:
            return False, str(e)

    def reset_history(self):
        self.history.clear()

    def get_action(self, px, py, entropy, pid, ult_ready, sub_ready):
        """현재 상태를 받아 다음 행동을 결정 (하이브리드 로직)"""
        if not self.lstm_model:
            return "None", "No Model"

        # 1. 데이터 전처리 (DataFrame -> Scaler)
        try:
            input_df = pd.DataFrame([[px, py, entropy, pid, ult_ready, sub_ready]], columns=self.feature_cols)
            feats_scaled = self.scaler.transform(input_df)
            self.history.append(feats_scaled[0])
        except Exception as e:
            print(f"Agent Data Error: {e}")
            return "None", "Error"

        action_name = "None"
        debug_msg = ""

        # 2. 결정 로직
        if len(self.history) == self.seq_length:
            # [A] LSTM 추론 (메인)
            inp = torch.FloatTensor(np.array([self.history])).to(self.device)
            with torch.no_grad():
                out = self.lstm_model(inp)
                _, pred = torch.max(out, 1)
                action_name = self.encoder.inverse_transform([pred.item()])[0]
            debug_msg = action_name
        else:
            # [B] Warm-up (RF or Random)
            if self.rf_model:
                # RF 입력 차원 맞추기
                n_features = getattr(self.rf_model, 'n_features_in_', 4)
                if n_features == 6:
                    rf_in = [[px, py, entropy, pid, ult_ready, sub_ready]]
                else:
                    rf_in = [[px, py, entropy, pid]]
                action_name = self.rf_model.predict(rf_in)[0]
                debug_msg = f"RF({len(self.history)})"
            else:
                action_name = random.choice(['left', 'right', 'space', 'None'])
                debug_msg = f"Wait({len(self.history)})"

        # 광클 방지
        if action_name == 'down': 
            action_name = "None"
            
        return action_name, debug_msg