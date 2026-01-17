# train_lstm.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# === 설정 ===
SEQ_LENGTH = 200  # 과거 10프레임을 보고 판단 (약 0.3~0.5초)
HIDDEN_SIZE = 256
NUM_LAYERS = 3
EPOCHS = 100      # 학습 횟수
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 학습 장치: {DEVICE} (4070 Super라면 'cuda'가 떠야 합니다)")

class MapleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(DEVICE)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        # 마지막 시퀀스의 결과만 사용
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train():
    # 1. 파일 선택
    root = tk.Tk(); root.withdraw()
    print("📂 학습할 CSV 데이터 파일을 선택하세요 (다중 선택 가능)...")
    files = filedialog.askopenfilenames(title="CSV 선택", filetypes=[("CSV", "*.csv")])
    if not files: return

    # 2. 데이터 로드 및 전처리
    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠️ 로드 실패 ({f}): {e}")
            
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    print(f"📊 총 데이터: {len(df)}개")

    # 노이즈 제거 (불필요한 키)
    ignore_keys = ['media_volume_up', 'esc', 'f1', 'alt_l', 'caps_lock']
    df = df[~df['key_pressed'].isin(ignore_keys)]

    df = df[df['key_pressed'] != 'down']

    none_df = df[df['key_pressed'] == 'None'].sample(frac=0.1, random_state=42)
    action_df = df[df['key_pressed'] != 'None']
    df = pd.concat([none_df, action_df])
    
    feature_cols = ['player_x', 'player_y', 'entropy', 'platform_id', 'ult_ready', 'sub_ready']
    
    # 결측치 처리 및 형변환
    for col in feature_cols:
        if col not in df.columns: df[col] = 0
    
    df[feature_cols] = df[feature_cols].fillna(0)
    df['key_pressed'] = df['key_pressed'].fillna('None').astype(str)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df['key_pressed'])

    # 3. 시퀀스 데이터 생성 (윈도우 슬라이딩)
    print("⏳ 시퀀스 데이터 변환 중... (잠시만 기다려주세요)")
    X_seq, y_seq = create_sequences(X_scaled, y_encoded, SEQ_LENGTH)
    
    # [핵심 수정] 샘플 수가 너무 적은(2개 미만) 클래스 제거
    unique, counts = np.unique(y_seq, return_counts=True)
    rare_classes = unique[counts < 2]
    
    if len(rare_classes) > 0:
        print(f"⚠️ [자동 보정] 샘플 수가 부족한 희귀 행동 {len(rare_classes)}종류를 학습에서 제외합니다.")
        # 희귀 클래스가 아닌 데이터만 남김
        mask = np.isin(y_seq, rare_classes, invert=True)
        X_seq = X_seq[mask]
        y_seq = y_seq[mask]

    # 4. 학습/검증 분리
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=True, stratify=y_seq)

    train_dataset = MapleDataset(X_train, y_train)
    test_dataset = MapleDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 모델 초기화
    num_classes = len(encoder.classes_)
    model = LSTMModel(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, num_classes).to(DEVICE)
    
    # 클래스 불균형 해결을 위한 가중치 계산 (옵션)
    # class_counts = torch.bincount(torch.tensor(y_train))
    # weights = 1. / (class_counts.float() + 1e-6)
    # criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. 학습 루프
    print(f"🔥 학습 시작 (Total Epochs: {EPOCHS})")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 검증
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")

    # 7. 저장
    save_path = "kinesis_lstm_model.pth"
    save_dict = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'encoder': encoder,
        'seq_length': SEQ_LENGTH,
        'input_size': len(feature_cols),
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_classes': num_classes,
        'feature_cols': feature_cols
    }
    
    torch.save(save_dict, save_path)
    print(f"💾 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    train()