import torch
import numpy as np
import pandas as pd
import joblib
from collections import deque
from modules.model import LSTMModel
from modules.rune_solver import PhysicsLearner
from modules.navigator import TacticalNavigator  # [필수] modules/navigator.py 생성 필요
from platform_manager import PlatformManager

class BotAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Agent Device: {self.device}")

        self.lstm_model = None
        self.rf_model = None
        self.scaler = None
        self.encoder = None
        
        self.action_queue = deque()
        self.seq_length = 10
        self.history = deque(maxlen=self.seq_length)
        
        # 기본 컬럼 정의 (로드 시 덮어써짐)
        self.feature_cols = ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready']

        # [신규] 자율 주행 및 전술 모듈
        self.pm = PlatformManager()
        self.physics = PhysicsLearner()
        # 물리 엔진 로드 (실패해도 치명적이지 않도록 try-except 처리 권장)
        try:
            self.physics.load_model("physics_hybrid_model.pth")
        except:
            print("⚠️ 물리 엔진 모델을 찾지 못했습니다. 네비게이션 성능이 저하될 수 있습니다.")

        self.navigator = TacticalNavigator(self.pm, self.physics)
        self.mode = "HYBRID" # HYBRID(LSTM+Nav) / AUTO(Nav Only)
        
        self.last_kill_count = 0

    def load_lstm(self, file_path):
        """LSTM 모델 및 메타데이터 로드"""
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.scaler = checkpoint['scaler']
            self.encoder = checkpoint['encoder']
            self.feature_cols = checkpoint.get('feature_cols', self.feature_cols)
            self.seq_length = checkpoint.get('seq_length', 10)
            
            # 모델 파라미터 로드
            input_size = checkpoint.get('input_size', len(self.feature_cols))
            hidden_size = checkpoint.get('hidden_size', 128)
            num_layers = checkpoint.get('num_layers', 2)
            num_classes = checkpoint.get('num_classes', 10)
            future_steps = checkpoint.get('future_steps', 1)
            dropout = checkpoint.get('dropout', 0)

            self.lstm_model = LSTMModel(
                input_size, hidden_size, num_layers, num_classes, future_steps, dropout
            ).to(self.device)
            
            self.lstm_model.load_state_dict(checkpoint['model_state'])
            self.lstm_model.eval()
            
            self.history = deque(maxlen=self.seq_length)
            return True, f"LSTM Loaded (Seq:{self.seq_length}, Future:{future_steps})"
        except Exception as e:
            return False, f"LSTM Error: {str(e)}"

    def load_rf(self, file_path):
        """Random Forest 모델 로드 (호환성 유지용)"""
        try:
            self.rf_model = joblib.load(file_path)
            return True, "RF Loaded"
        except Exception as e:
            return False, str(e)

    def reset_history(self):
        self.history.clear()
        self.action_queue.clear()
        self.last_kill_count = 0

    def on_map_change(self, map_json_path):
        """맵 변경 시 네비게이터 재설정"""
        self.pm.load_platforms(map_json_path)
        self.navigator.build_graph()

    def get_action(self, px, py, entropy, pid, ult_ready, sub_ready, dist_left=0, dist_right=0, current_kill_count=0):
        """
        봇의 행동 결정 (LSTM + Tactical Navigator)
        """
        # 1. 킬 보상 업데이트 (네비게이터에게 정보 제공)
        kill_diff = max(0, current_kill_count - self.last_kill_count)
        self.last_kill_count = current_kill_count
        
        if kill_diff > 0:
            self.navigator.update_combat_stats(px, py, kill_diff)

        # 2. 큐 확인 (이미 계획된 행동 수행)
        if self.action_queue:
            return self.action_queue.popleft(), f"Seq({len(self.action_queue)})"

        # 3. LSTM 추론 준비
        lstm_action = "None"
        lstm_status = "Wait"
        
        if self.lstm_model:
            try:
                # 데이터 전처리
                input_data = {
                    'player_x': px, 'player_y': py, 'entropy': entropy, 
                    'platform_id': pid, 'ult_ready': ult_ready, 'sub_ready': sub_ready,
                    'dist_left': dist_left, 'dist_right': dist_right, # gui.py에서 넘겨주는 거리
                    # 아래 값들은 기본값 0 (gui.py에서 계산 안 하므로)
                    'inv_dist_up': 0, 'inv_dist_down': 0, 'inv_dist_left': 0, 'inv_dist_right': 0,
                    'corner_tl': 0, 'corner_tr': 0, 'corner_bl': 0, 'corner_br': 0
                }
                
                df = pd.DataFrame([input_data])
                for col in self.feature_cols:
                    if col not in df.columns: df[col] = 0
                
                feats_scaled = self.scaler.transform(df[self.feature_cols])
                self.history.append(feats_scaled[0])

                if len(self.history) == self.seq_length:
                    inp = torch.FloatTensor(np.array([self.history])).to(self.device)
                    with torch.no_grad():
                        out = self.lstm_model(inp) # Output: (1, Future, Classes)
                        _, preds = torch.max(out, 2)
                        preds = preds.squeeze(0).cpu().numpy() # (Future,)
                        
                        # 미래 예측 행동들을 큐에 담음
                        actions = self.encoder.inverse_transform(preds)
                        self.action_queue.extend(actions)
                        
                        lstm_action = self.action_queue.popleft()
                        lstm_status = "LSTM"
            except Exception as e:
                print(f"Agent Action Error: {e}")

        # 4. [핵심] 하이브리드 판단 (네비게이터 개입)
        nav_action, nav_msg = self.navigator.get_move_decision(px, py)
        
        # A. 캠핑 모드일 때 (명당 자리 사수)
        if "Camping" in nav_msg:
            # 꿀자리에 있으므로 이동(Left/Right)은 자제하고, 공격/설치기 위주로 수행
            if lstm_action != "None" and ("left" in lstm_action or "right" in lstm_action):
                # LSTM이 이동하려고 하면 무시 (캠핑 유지)
                return "None", "Camping(Hold)"
            
            # 공격이나 스킬 사용이면 LSTM 따름 (아니면 랜덤 공격)
            if lstm_action != "None":
                return lstm_action, "Camping(Act)"
            else:
                # 할 게 없으면 네비게이터가 'None'을 줘서 대기하거나, 광역기 쿨타임 체크 후 사용
                if sub_ready == 1: return "q", "Camp+Atk" # 예: Q가 광역기라면
                return "None", "Camping"

        # B. 탐색/이동 모드일 때
        # LSTM이 멍때리거나(None), 확신이 없거나, AUTO 모드일 때 네비게이터가 길 안내
        if self.mode == "AUTO" or lstm_action == "None" or not self.lstm_model:
            if nav_action != "None":
                # 이동하면서 공격 섞기 (Nav+Atk)
                if sub_ready == 1: 
                    return f"{nav_action}+q", "Nav+Atk"
                return nav_action, nav_msg
            
        return lstm_action, lstm_status