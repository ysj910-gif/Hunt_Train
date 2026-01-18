import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os
import json

# === [1] 모델 클래스 정의 ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, future_steps=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.future_steps = future_steps
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes * future_steps)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.reshape(-1, self.future_steps, self.num_classes)

# === [2] 하이퍼파라미터 설정 ===
SEQ_LENGTH = 150       # 과거 150프레임(약 4~5초)을 보고 판단
FUTURE_STEPS = 30      # 미래 30프레임(약 1초) 예측
HIDDEN_SIZE = 256      # 모델 용량
NUM_LAYERS = 4         # 레이어 깊이
DROPOUT = 0.3
EPOCHS = 300
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.98           # 보상 감가율 (미래의 킬을 현재 가치로 환산할 때 사용)

# 학습에 사용할 특성 컬럼 (upgrade_data.py 결과물)
FEATURE_COLS = [
    'player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready',
    'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right',
    'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
]
TARGET_COL = 'key_pressed'

# === [3] 설정 파일 로드 (설치기 인식용) ===
def load_install_skills():
    """hunter_config.json에서 지속시간(dur)이 2초 이상인 스킬을 설치기로 인식"""
    config_path = "hunter_config.json"
    install_skills = {} # { 'key': duration }
    
    if not os.path.exists(config_path):
        print("⚠️ 설정 파일(hunter_config.json)이 없습니다. 설치기 학습 기능을 건너뜁니다.")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            mapping = data.get("mapping", {})
            
            print("\n🔧 [스킬 설정 로드]")
            for name, info in mapping.items():
                key = info.get("key", "").lower()
                dur = float(info.get("dur", 0))
                
                # 지속시간이 2.0초 이상이면 설치기로 간주
                if key and dur >= 2.0:
                    install_skills[key] = dur
                    print(f"   - 설치기 감지: [{key.upper()}] {name} (지속 {dur}초)")
                    
        return install_skills
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return {}

# === [4] 스마트 보상 계산 (Elite Data Filtering 핵심) ===
def calculate_smart_rewards(df, install_skills, gamma=0.98):
    """
    1. 일반 공격: 킬이 발생하면 그 직전 행동들에 점수 부여 (Backpropagation)
    2. 설치기: 설치 후 지속시간 동안 발생한 총 킬 수를 합산하여 점수 부여
    """
    timestamps = df['timestamp'].values
    actions = df['key_pressed'].fillna('None').astype(str).values
    
    # kill_reward 컬럼 확인 및 생성
    if 'kill_reward' not in df.columns:
        if 'kill_count' in df.columns:
            # kill_count의 변화량으로 reward 계산 (음수 제거)
            rewards = df['kill_count'].diff().fillna(0).values
            rewards[rewards < 0] = 0
        else:
            rewards = np.zeros(len(df))
    else:
        rewards = df['kill_reward'].values

    # [1] 기본 단기 보상 (일반 공격용)
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    # 뒤에서부터 앞으로 계산 (나중에 잡은 킬 점수를 앞쪽 행동에 나눠줌)
    for t in reversed(range(len(rewards))):
        if rewards[t] > 0:
            running_add = rewards[t]
        else:
            running_add = running_add * gamma
        discounted[t] = running_add

    # [2] 설치기 장기 보상 (설정 파일 기반)
    if install_skills:
        for t in range(len(df)):
            action = actions[t].lower()
            
            # 현재 누른 키가 설치기에 포함되는지 확인 (예: 'down+e' -> 'e')
            matched_dur = 0
            for k, dur in install_skills.items():
                if k in action: 
                    matched_dur = dur
                    break
            
            if matched_dur > 0:
                current_time = timestamps[t]
                future_kills = 0
                
                # 설치기 지속시간 동안 미래의 킬을 미리 내다봄
                for future_t in range(t + 1, len(df)):
                    if timestamps[future_t] - current_time > matched_dur:
                        break
                    future_kills += rewards[future_t]
                
                # 보상 정책: 설치기 하나로 3마리 이상 잡아야 이득
                if future_kills >= 3: 
                    bonus = future_kills * 3.0 # 강력한 보너스
                    discounted[t] += bonus
                else:
                    discounted[t] -= 5.0 # 낭비 시 강력한 패널티 (쓰지 마!)
    
    df['discounted_reward'] = discounted
    return df

# === [5] 시퀀스 생성 (데이터 선별) ===
def create_sequences_smart(df, seq_length, future_steps, scaler, encoder):
    # 특성 컬럼 채우기
    for col in FEATURE_COLS:
        if col not in df.columns: df[col] = 0
            
    data_scaled = scaler.transform(df[FEATURE_COLS])
    target_values = encoder.transform(df[TARGET_COL].astype(str).values)
    values = df['discounted_reward'].values
    
    xs, ys = [], []
    for i in range(len(df) - seq_length - future_steps + 1):
        target_idx = i + seq_length
        
        # [핵심 필터링]
        # 해당 시점의 행동 가치(Reward)가 너무 낮으면(0.01 이하) -> 쓸모없는 행동
        # 쓸모없는 행동은 90% 확률로 학습 데이터에서 제외 (과감한 삭제)
        if values[target_idx] <= 0.01 and np.random.rand() > 0.1:
            continue
            
        x_window = data_scaled[i : i + seq_length]
        y_window = target_values[i + seq_length : i + seq_length + future_steps]
        xs.append(x_window)
        ys.append(y_window)
        
    return np.array(xs), np.array(ys)

# === [6] 메인 학습 함수 ===
def train():
    root = tk.Tk(); root.withdraw()
    
    # 1. 설정 및 데이터 로드
    install_skills = load_install_skills()
    
    print("\n📂 학습할 CSV 데이터 파일들을 선택하세요 (upgrade_data.py 변환 파일 권장)...")
    files = filedialog.askopenfilenames(title="CSV 선택", filetypes=[("CSV", "*.csv")])
    if not files: return

    print("⏳ 데이터 로드 및 스마트 보상 계산 중...")
    temp_dfs = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # 노이즈 키 제거
            ignore_keys = ['media_volume_up', 'esc', 'f1', 'alt_l', 'caps_lock', 'unknown']
            df = df[~df['key_pressed'].isin(ignore_keys)]
            df['key_pressed'] = df['key_pressed'].fillna('None')
            df['platform_id'] = df['platform_id'].fillna(-1)
            
            # [보상 계산] 여기서 '잘한 행동'에 점수를 매깁니다.
            df = calculate_smart_rewards(df, install_skills, gamma=GAMMA)
            
            # (디버그) 최대 보상 점수 출력
            max_r = df['discounted_reward'].max()
            print(f"   - {os.path.basename(f)}: 최대 가치 점수 {max_r:.2f}")
            
            temp_dfs.append(df)
        except Exception as e:
            print(f"⚠️ 로드 실패 ({os.path.basename(f)}): {e}")
            
    if not temp_dfs: return
    full_df = pd.concat(temp_dfs, ignore_index=True)
    
    # 2. 스케일러 & 인코더 학습
    scaler = StandardScaler()
    scaler.fit(full_df[FEATURE_COLS])
    encoder = LabelEncoder()
    encoder.fit(full_df[TARGET_COL].astype(str))
    num_classes = len(encoder.classes_)

    # 3. 시퀀스 데이터 생성 (필터링 적용)
    print(f"✂️ 의미 없는 구간(Idle) 제거 및 학습 데이터 생성 중...")
    X_list, y_list = [], []
    for df in temp_dfs:
        xs, ys = create_sequences_smart(df, SEQ_LENGTH, FUTURE_STEPS, scaler, encoder)
        if len(xs) > 0:
            X_list.append(xs)
            y_list.append(ys)
            
    if not X_list: print("❌ 학습 데이터 부족 (모든 데이터가 필터링됨)"); return

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"✨ 최종 학습 데이터: {len(X_all)}개 시퀀스 (사냥 효율 최적화됨)")

    # 4. 학습 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True)
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE))
    test_dataset = TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 모델 생성
    model = LSTMModel(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, num_classes, FUTURE_STEPS, DROPOUT).to(DEVICE)
    
    # [가중치 적용] 'None' 클래스는 점수를 깎아서 더 적극적으로 움직이게 유도
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_all.flatten()), y=y_all.flatten())
    try:
        none_idx = encoder.transform(['None'])[0]
        class_weights[none_idx] *= 0.1 # None 가중치 1/10 토막
        print(f"🔥 'None' 클래스 패널티 적용됨 (적극성 강화)")
    except: pass
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 6. 학습 루프
    print(f"\n🔥 엘리트 학습 시작...")
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs.view(-1, num_classes), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                outputs = model(bx)
                _, predicted = torch.max(outputs, 2)
                correct += (predicted == by).sum().item()
                total += by.numel()
        
        acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% (⭐ Best)")
        elif (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # 7. 모델 저장
    save_path = "kinesis_lstm_best.pth"
    torch.save({
        'model_state': best_model_state,
        'scaler': scaler, 'encoder': encoder,
        'feature_cols': FEATURE_COLS,
        'input_size': len(FEATURE_COLS), 'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS, 'num_classes': num_classes,
        'seq_length': SEQ_LENGTH, 'future_steps': FUTURE_STEPS, 'dropout': DROPOUT
    }, save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    train()