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
        
        # 시퀀스 큐 (미래 행동 계획)
        self.action_queue = deque()
        
        # 기억 저장소 (LSTM용)
        self.seq_length = 10
        self.history = deque(maxlen=self.seq_length)
        
        # 입력 특성 컬럼 (기본값)
        self.feature_cols = ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready']

    def load_lstm(self, file_path):
        """LSTM 모델 로드"""
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            
            self.scaler = checkpoint['scaler']
            self.encoder = checkpoint['encoder']
            self.feature_cols = checkpoint.get('feature_cols', self.feature_cols)
            
            # 시퀀스 길이 업데이트
            self.seq_length = checkpoint.get('seq_length', 10)
            self.history = deque(maxlen=self.seq_length)
            
            # 모델 생성
            self.lstm_model = LSTMModel(
                input_size=checkpoint.get('input_size', 6),
                hidden_size=checkpoint.get('hidden_size', 128),
                num_layers=checkpoint.get('num_layers', 2),
                num_classes=checkpoint.get('num_classes', 10),
                future_steps=checkpoint.get('future_steps', 1),
                dropout=checkpoint.get('dropout', 0)
            ).to(self.device)
            
            self.lstm_model.load_state_dict(checkpoint['model_state'])
            self.lstm_model.eval()
            return True, f"LSTM Loaded (Seq: {self.seq_length})"
        except Exception as e:
            return False, str(e)

    # [★추가] 이 함수가 없어서 에러가 났었습니다.
    def load_rf(self, file_path):
        """Random Forest 모델 로드"""
        try:
            self.rf_model = joblib.load(file_path)
            return True, "RF Loaded"
        except Exception as e:
            return False, str(e)

    def reset_history(self):
        self.history.clear()
        self.action_queue.clear()

    def get_action(self, px, py, entropy, pid, ult_ready, sub_ready, dist_left=0, dist_right=0):
        """현재 상태를 받아 다음 행동을 결정"""
        
        # 1. 큐에 계획된 행동이 있으면 즉시 반환
        if self.action_queue:
            return self.action_queue.popleft(), f"Seq({len(self.action_queue)})"

        if not self.lstm_model:
            return "None", "No Model"

        # 2. 데이터 전처리
        try:
            input_data = {
                'player_x': px, 'player_y': py, 'entropy': entropy, 
                'platform_id': pid, 'ult_ready': ult_ready, 'sub_ready': sub_ready,
                'dist_left': dist_left, 'dist_right': dist_right,
                # 만약 학습 때 inv_dist 등 고급 특성을 썼다면 여기서도 계산해서 넣어줘야 함 (간소화를 위해 생략 가능하나 성능 영향 있음)
                # 여기서는 일단 0으로 채워서 에러 방지
                'inv_dist_up': 0, 'inv_dist_down': 0, 'inv_dist_left': 0, 'inv_dist_right': 0,
                'corner_tl': 0, 'corner_tr': 0, 'corner_bl': 0, 'corner_br': 0
            }
            
            df = pd.DataFrame([input_data])
            
            # 컬럼 순서 맞추기 & 없는 컬럼 0 채우기
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # 스케일링
            feats_scaled = self.scaler.transform(df[self.feature_cols])
            self.history.append(feats_scaled[0])
            
        except Exception as e:
            print(f"Agent Data Error: {e}")
            return "None", "Error"

        action_name = "None"
        debug_msg = ""

        # 3. 결정 로직
        if len(self.history) == self.seq_length:
            inp = torch.FloatTensor(np.array([self.history])).to(self.device)
            with torch.no_grad():
                out = self.lstm_model(inp)
                _, preds = torch.max(out, 2)
                preds = preds.squeeze(0).cpu().numpy()
                
                actions = self.encoder.inverse_transform(preds)
                self.action_queue.extend(actions)
                
                action_name = self.action_queue.popleft()
                debug_msg = "LSTM(New)"
        else:
            action_name = "None"
            debug_msg = f"Wait({len(self.history)})"

        return action_name, debug_msg