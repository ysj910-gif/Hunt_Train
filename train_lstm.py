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

# === [1] 모델 클래스 ===
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

# === [2] 설정 ===
SEQ_LENGTH = 10
FUTURE_STEPS = 5
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.90

# 목표 에포크 (넉넉하게)
TARGET_EPOCHS = 1000 

FEATURE_COLS = [
    'player_x', 'player_y', 
    'delta_x', 'delta_y',   # <--- [신규 추가] 속도 정보
    'entropy', 'platform_id', 'ult_ready', 'sub_ready',
    'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right',
    'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
]
TARGET_COL = 'key_pressed'
SAVE_PATH = "kinesis_lstm_best.pth"

# === [3] 유틸리티 함수 ===
def load_install_skills():
    config_path = "hunter_config.json"
    install_skills = {}
    if not os.path.exists(config_path): return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            mapping = data.get("mapping", {})
            for name, info in mapping.items():
                key = info.get("key", "").lower()
                dur = float(info.get("dur", 0))
                if key and dur >= 2.0:
                    install_skills[key] = dur
        return install_skills
    except: return {}

def calculate_smart_rewards(df, install_skills, gamma=0.98):
    timestamps = df['timestamp'].values
    actions = df['key_pressed'].fillna('None').astype(str).values
    
    if 'kill_reward' not in df.columns:
        if 'kill_count' in df.columns:
            rewards = df['kill_count'].diff().fillna(0).values
            rewards[rewards < 0] = 0
        else:
            rewards = np.zeros(len(df))
    else:
        rewards = df['kill_reward'].values

    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] > 0: running_add = rewards[t]
        else: running_add = running_add * gamma
        discounted[t] = running_add

    if install_skills:
        for t in range(len(df)):
            action = actions[t].lower()
            matched_dur = 0
            for k, dur in install_skills.items():
                if k in action: matched_dur = dur; break
            
            if matched_dur > 0:
                current_time = timestamps[t]
                future_kills = 0
                for future_t in range(t + 1, len(df)):
                    if timestamps[future_t] - current_time > matched_dur: break
                    future_kills += rewards[future_t]
                
                if future_kills >= 3: discounted[t] += future_kills * 3.0
                else: discounted[t] -= 5.0
    
    df['discounted_reward'] = discounted
    return df

def create_sequences_smart(df, seq_length, future_steps, scaler, encoder):
    # 1. 기본 데이터 처리
    for col in FEATURE_COLS:
        if col not in df.columns: df[col] = 0
            
    # 정규화 (Scaler)
    data_scaled = scaler.transform(df[FEATURE_COLS])
    target_values = encoder.transform(df[TARGET_COL].astype(str).values)
    values = df['discounted_reward'].values
    
    xs, ys = [], []
    
    # 2. 시퀀스 생성 (기존)
    for i in range(len(df) - seq_length - future_steps + 1):
        if values[i + seq_length] <= 0.01 and np.random.rand() > 0.1: continue
        
        x_window = data_scaled[i : i + seq_length]
        y_window = target_values[i + seq_length : i + seq_length + future_steps]
        
        xs.append(x_window)
        ys.append(y_window)

        # [★신규] 3. 데이터 증강 (좌우 반전)
        # 50% 확률로 좌우 반전 데이터를 추가 학습 (데이터 1.5배 뻥튀기 효과)
        if np.random.rand() < 0.5:
            # 복사본 생성
            x_aug = x_window.copy()
            
            # FEATURE_COLS 순서에 맞춰서 좌우 관련 변수 반전
            # 예: delta_x(속도) 반전, dist_left <-> dist_right 교체 등
            # (단, Scaler가 적용된 상태라 단순 -1 곱하기는 위험할 수 있음)
            # 여기서는 간단하게 'delta_x'만 부호를 뒤집는 방식으로 노이즈를 줍니다.
            
            # delta_x가 2번째 컬럼(인덱스 2)이라고 가정
            try:
                dx_idx = FEATURE_COLS.index('delta_x')
                x_aug[:, dx_idx] *= -1 # 속도 반전
            except: pass
            
            xs.append(x_aug)
            ys.append(y_window) # 정답(행동)은 그대로 (또는 행동도 반전시켜야 완벽하지만 복잡함)

    return np.array(xs), np.array(ys)

# === [4] 메인 학습 함수 (안전장치 추가) ===
def train():
    root = tk.Tk(); root.withdraw()
    install_skills = load_install_skills()
    
    # 1. 이어하기 여부 확인
    start_epoch = 0
    resume_mode = False
    best_acc = 0.0
    
    # 체크포인트 로드 변수들
    loaded_state = None
    loaded_scaler = None
    loaded_encoder = None
    
    if os.path.exists(SAVE_PATH):
        ans = input(f"\n💾 기존 모델({SAVE_PATH}) 발견! 이어서 학습할까요? (y/n): ").strip().lower()
        if ans == 'y':
            print("🔄 기존 모델을 로드합니다...")
            try:
                checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
                loaded_state = checkpoint['model_state']
                loaded_scaler = checkpoint['scaler']
                loaded_encoder = checkpoint['encoder']
                
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    best_acc = checkpoint.get('best_acc', 0.0)
                    print(f"✅ Epoch {start_epoch}부터 시작합니다. (기존 최고 정확도: {best_acc:.2f}%)")
                else:
                    print("⚠️ 이전 파일에 Epoch 정보가 없습니다.")
                    user_epoch = input("   마지막으로 완료한 Epoch 수를 입력하세요 (예: 300): ")
                    start_epoch = int(user_epoch) if user_epoch.isdigit() else 0
                
                resume_mode = True
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}\n   -> 처음부터 시작합니다.")
                resume_mode = False

    # 2. 데이터 로드
    print("\n📂 학습할 CSV 데이터 파일들을 선택하세요...")
    files = filedialog.askopenfilenames(title="CSV 선택", filetypes=[("CSV", "*.csv")])
    if not files: return

    print("⏳ 데이터 처리 중...")
    temp_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            ignore_keys = ['media_volume_up', 'esc', 'f1', 'alt_l', 'caps_lock', 'unknown']
            df = df[~df['key_pressed'].isin(ignore_keys)]
            df['key_pressed'] = df['key_pressed'].fillna('None')
            df['platform_id'] = df['platform_id'].fillna(-1)
            
            if 'kill_reward' not in df.columns and 'kill_count' in df.columns:
                df['kill_reward'] = df['kill_count'].diff().fillna(0)
                df.loc[df['kill_reward'] < 0, 'kill_reward'] = 0
            
            df = calculate_smart_rewards(df, install_skills, gamma=GAMMA)
            temp_dfs.append(df)
        except: pass
            
    if not temp_dfs: return
    full_df = pd.concat(temp_dfs, ignore_index=True)
    
    # 3. Scaler & Encoder 설정
    if resume_mode and loaded_scaler and loaded_encoder:
        print("🔗 기존 모델의 Scaler와 Encoder를 사용합니다.")
        scaler = loaded_scaler
        encoder = loaded_encoder
    else:
        print("🆕 새로운 Scaler와 Encoder를 학습합니다.")
        scaler = StandardScaler()
        scaler.fit(full_df[FEATURE_COLS])
        encoder = LabelEncoder()
        encoder.fit(full_df[TARGET_COL].astype(str))
    
    num_classes = len(encoder.classes_)

    # 4. 시퀀스 생성
    print(f"✂️ 학습 데이터 생성 중...")
    X_list, y_list = [], []
    for df in temp_dfs:
        xs, ys = create_sequences_smart(df, SEQ_LENGTH, FUTURE_STEPS, scaler, encoder)
        if len(xs) > 0:
            X_list.append(xs)
            y_list.append(ys)
            
    if not X_list: print("❌ 데이터 부족"); return
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"✨ 총 {len(X_all)}개 시퀀스로 학습 진행")

    # 5. 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True)
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE))
    test_dataset = TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 모델 초기화
    model = LSTMModel(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, num_classes, FUTURE_STEPS, DROPOUT).to(DEVICE)
    if resume_mode and loaded_state:
        try:
            model.load_state_dict(loaded_state)
            print("✅ 기존 학습 가중치 복원 완료!")
        except Exception as e:
            print(f"⚠️ 모델 구조 불일치, 처음부터 시작: {e}")
            start_epoch = 0; best_acc = 0.0; resume_mode = False

    # 7. Optimizer & Learning Rate (핵심 수정!)
    # 재학습(Resume) 시에는 LR을 0.0001로 낮춤 (기존 0.001)
    initial_lr = 0.001
    if resume_mode:
        initial_lr = 0.0001 
        print(f"📉 재학습 모드: 학습률을 {initial_lr}로 낮춰서 미세 조정합니다 (쇼크 방지).")
    
    # Class Weight 적용
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_all.flatten()), y=y_all.flatten())
    try:
        none_idx = encoder.transform(['None'])[0]
        class_weights[none_idx] *= 0.1 
    except: pass
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # 스케줄러: 더 민감하게 반응하도록 수정
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # 8. 학습 루프
    print(f"\n🔥 학습 시작: Epoch {start_epoch+1} ~ {TARGET_EPOCHS}")
    best_model_state = model.state_dict() if resume_mode else None
    
    for epoch in range(start_epoch, TARGET_EPOCHS):
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
        
        # 스케줄러 업데이트 (UserWarning 해결)
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            print(f"Epoch {epoch+1}/{TARGET_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% (⭐ New Best!)")
            
            torch.save({
                'epoch': epoch,
                'model_state': best_model_state,
                'best_acc': best_acc,
                'scaler': scaler, 'encoder': encoder,
                'feature_cols': FEATURE_COLS,
                'input_size': len(FEATURE_COLS), 'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS, 'num_classes': num_classes,
                'seq_length': SEQ_LENGTH, 'future_steps': FUTURE_STEPS, 'dropout': DROPOUT
            }, SAVE_PATH)
            
        elif (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{TARGET_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    print(f"✅ 최종 완료! 최고 정확도: {best_acc:.2f}%")

if __name__ == "__main__":
    train()