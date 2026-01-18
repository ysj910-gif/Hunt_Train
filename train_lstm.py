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
import joblib
import os

# === [1] 모델 클래스 정의 (Dropout + Future Steps + 버그 수정) ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, future_steps=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.future_steps = future_steps
        self.num_classes = num_classes  # [수정] 누락되었던 변수 추가
        
        # LSTM 레이어 (Deep Structure + Dropout)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 출력층: 미래의 N개 행동을 모두 예측 (Many-to-Many 형태)
        self.fc = nn.Linear(hidden_size, num_classes * future_steps)

    def forward(self, x):
        # 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 실행
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 타임스텝의 히든 스테이트만 사용하여 미래 전체를 예측
        out = self.fc(out[:, -1, :])
        
        # (Batch, Future_Steps, Num_Classes) 형태로 변환
        return out.reshape(-1, self.future_steps, self.num_classes)

# === [2] 하이퍼파라미터 설정 ===
SEQ_LENGTH = 200      # 과거 200프레임(약 6초)을 보고 판단 (Long Term Memory)
FUTURE_STEPS = 30     # 미래 30프레임(약 1초)의 행동을 미리 계획
HIDDEN_SIZE = 256     # 뇌 용량 (High Capacity)
NUM_LAYERS = 4        # 4층 구조 (Very Deep Learning)
DROPOUT = 0.3         # 과적합 방지 (30% 망각)
EPOCHS = 500          # 학습 횟수
BATCH_SIZE = 64       # 배치 크기
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [핵심] 학습에 사용할 특성 (upgrade_data.py 결과물)
FEATURE_COLS = [
    'player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready',
    # 위기 감지 센서 (거리 역수)
    'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right',
    # 네비게이션 센서 (모서리 거리)
    'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
]
TARGET_COL = 'key_pressed'

# === [3] 시퀀스 생성 함수 ===
def create_sequences(df, seq_length, future_steps, scaler, encoder):
    # 특성 컬럼이 없으면 0으로 채움 (호환성 유지)
    for col in FEATURE_COLS:
        if col not in df.columns: df[col] = 0
            
    # 데이터 스케일링 (DataFrame 형태 유지하여 경고 방지)
    data_scaled = scaler.transform(df[FEATURE_COLS])
    
    # 타겟 인코딩
    target_values = encoder.transform(df[TARGET_COL].astype(str).values)
    
    xs, ys = [], []
    # 데이터가 너무 짧으면 스킵
    if len(df) <= seq_length + future_steps:
        return np.array([]), np.array([])

    # Sliding Window 방식으로 데이터 생성
    for i in range(len(df) - seq_length - future_steps + 1):
        x_window = data_scaled[i : i + seq_length]
        y_window = target_values[i + seq_length : i + seq_length + future_steps]
        xs.append(x_window)
        ys.append(y_window)
        
    return np.array(xs), np.array(ys)

# === [4] 메인 학습 함수 ===
def train():
    # 1. 파일 선택
    root = tk.Tk(); root.withdraw()
    print("📂 학습할 CSV 데이터 파일들을 선택하세요 (upgrade_data.py로 변환된 파일 권장)...")
    files = filedialog.askopenfilenames(title="CSV 선택", filetypes=[("CSV", "*.csv")])
    if not files: return

    # 2. 데이터 로드 및 병합
    print("⏳ 데이터 로드 중...")
    temp_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # 노이즈 데이터 제거
            ignore_keys = ['media_volume_up', 'esc', 'f1', 'alt_l', 'caps_lock', 'unknown']
            df = df[~df['key_pressed'].isin(ignore_keys)]
            
            # 결측치 처리
            df['key_pressed'] = df['key_pressed'].fillna('None')
            df['platform_id'] = df['platform_id'].fillna(-1)
            
            temp_dfs.append(df)
        except Exception as e:
            print(f"⚠️ 로드 실패 ({os.path.basename(f)}): {e}")
            
    if not temp_dfs: 
        print("❌ 유효한 데이터가 없습니다."); return
        
    full_df = pd.concat(temp_dfs, ignore_index=True)
    
    # 3. Scaler & Encoder 학습
    print("⚖️ 데이터 스케일링 (특이값 보정) 중...")
    # 전체 데이터셋 기준으로 스케일러 학습
    for col in FEATURE_COLS:
        if col not in full_df.columns: full_df[col] = 0
            
    scaler = StandardScaler()
    scaler.fit(full_df[FEATURE_COLS])
    
    encoder = LabelEncoder()
    encoder.fit(full_df[TARGET_COL].astype(str))
    
    num_classes = len(encoder.classes_)
    print(f"🏷️ 클래스: {num_classes}개, 특성: {len(FEATURE_COLS)}개 (고급 거리 센서 포함)")

    # 4. 시퀀스 변환 (메모리 효율 고려)
    print(f"✂️ 시퀀스 변환 중 (Seq: {SEQ_LENGTH}, Future: {FUTURE_STEPS})...")
    X_list, y_list = [], []
    
    for df in temp_dfs:
        xs, ys = create_sequences(df, SEQ_LENGTH, FUTURE_STEPS, scaler, encoder)
        if len(xs) > 0:
            X_list.append(xs)
            y_list.append(ys)
            
    if not X_list:
        print("❌ 학습 가능한 시퀀스 데이터가 부족합니다.")
        return

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    # 5. 데이터셋 준비 (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=True, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE))
    test_dataset = TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 모델 생성 및 설정
    model = LSTMModel(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        future_steps=FUTURE_STEPS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # [안정화] 학습률 스케줄러 (성능 정체 시 학습률 감소)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 7. 학습 루프
    print(f"🔥 학습 시작 (Device: {DEVICE})")
    
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            outputs = model(bx) # (Batch, Future, Classes)
            
            # Loss 계산: (Batch * Future, Classes) 형태로 펼쳐서 계산
            loss = criterion(outputs.view(-1, num_classes), by.view(-1))
            
            loss.backward()
            
            # [안정화] Gradient Clipping (Loss 폭발 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        # 검증 (Validation)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                outputs = model(bx) # (Batch, Future, Classes)
                _, predicted = torch.max(outputs, 2) # (Batch, Future)
                
                # 모든 타임스텝의 예측이 맞는지 확인 (전체 정확도)
                correct += (predicted == by).sum().item()
                total += by.numel() # Batch * Future
        
        acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        
        # 스케줄러 업데이트
        scheduler.step(acc)

        # 최고 기록 저장
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% (⭐ New Best!)")
        else:
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    # 8. 저장 (최고 성능 모델)
    print(f"\n💾 최고 정확도({best_acc:.2f}%) 모델 저장 중...")
    
    save_path = "kinesis_lstm_best.pth"
    
    # 저장할 모든 메타데이터 포함
    save_dict = {
        'model_state': best_model_state,
        'scaler': scaler,
        'encoder': encoder,
        'feature_cols': FEATURE_COLS,
        'input_size': len(FEATURE_COLS),
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_classes': num_classes,
        'seq_length': SEQ_LENGTH,
        'future_steps': FUTURE_STEPS,
        'dropout': DROPOUT
    }
    
    torch.save(save_dict, save_path)
    print(f"✅ 저장 완료: {save_path}")

if __name__ == "__main__":
    train()