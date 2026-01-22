import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing

# ---------------------------------------------------------
# 🚀 RTX 4070 Super 성능 최적화 설정
# ---------------------------------------------------------
# 배치 사이즈: 64 -> 2048 (VRAM 빵빵하니까 크게 잡아서 GPU 갈구기)
BATCH_SIZE = 2048  
# 데이터 로더 병렬 처리: CPU 코어 활용 (보통 코어 수의 절반)
NUM_WORKERS = min(8, multiprocessing.cpu_count())
# 혼합 정밀도(FP16): 텐서 코어 활용 (속도 2배, 메모리 절약)
USE_AMP = True 

# 학습 설정
SEQ_LENGTH = 10
EPOCHS = 300       # 속도가 빠르니 에폭을 늘려도 됨
LEARNING_RATE = 0.002 # 배치가 커지면 학습률도 살짝 올려야 함
PATIENCE = 20
MODEL_SAVE_PATH = "best_gru_model_boost.pth"

# ---------------------------------------------------------

from modules.model import LSTMModel 

FEATURE_COLS = [
    'player_x', 'player_y', 'delta_x', 'delta_y', 
    'entropy', 'platform_id', 'ult_ready', 'sub_ready', 
    'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right', 
    'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
]

# CuDNN 벤치마킹 활성화 (고정된 입력 크기에서 속도 최적화)
torch.backends.cudnn.benchmark = True

class MapleDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i+seq_length]
        y = labels[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train():
    print(f"🚀 High-Performance Training Mode")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: {USE_AMP}")
    
    # 1. 데이터 로드
    files = glob.glob("upgraded_*.csv")
    if not files:
        print("❌ 데이터 파일이 없습니다.")
        return

    df_list = []
    print("📂 CSV 파일 로딩 중...")
    for f in tqdm(files):
        try:
            d = pd.read_csv(f)
            if all(col in d.columns for col in FEATURE_COLS) and 'key_pressed' in d.columns:
                df_list.append(d)
        except: pass
    
    full_df = pd.concat(df_list, ignore_index=True).fillna(0)
    print(f"✅ 총 데이터: {len(full_df)} rows")

    # 2. 전처리
    scaler = MinMaxScaler()
    encoder = LabelEncoder()
    X_data = scaler.fit_transform(full_df[FEATURE_COLS])
    y_data = encoder.fit_transform(full_df['key_pressed'].astype(str))
    
    X_seq, y_seq = create_sequences(X_data, y_data, SEQ_LENGTH)
    
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=True, random_state=42)

    # pin_memory=True: CPU -> GPU 전송 속도 향상
    train_dataset = MapleDataset(X_train, y_train)
    val_dataset = MapleDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True # 워커 프로세스 유지
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )

    # 3. 모델 설정
    device = torch.device("cuda")
    input_size = len(FEATURE_COLS)
    num_classes = len(encoder.classes_)
    
    model = LSTMModel(
        input_size, 
        hidden_size = 256, 
        num_layers = 3, 
        num_classes = num_classes, 
        dropout=0.4
    ).to(device)
    
    # [최적화] PyTorch 2.0 컴파일 (가능하면 적용)
    try:
        model = torch.compile(model)
        print("⚡ Torch.compile 적용 완료 (속도 향상)")
    except: pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # AMP Scaler
    scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 4. 학습 루프
    best_loss = float('inf')
    patience_curr = 0
    train_hist, val_hist = [], []

    print("\n🔥 학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0
        
        # TQDM으로 진행상황 표시
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scaled Backward
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            run_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
        avg_train_loss = run_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_hist.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_hist.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} ({train_acc:.1f}%) | Val Loss={avg_val_loss:.4f} ({val_acc:.1f}%)")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_curr = 0
            # 저장
            save_dict = {
                'model_state': model.state_dict(), # compile된 모델은 unwrap 필요할 수 있음
                'scaler': scaler,
                'encoder': encoder,
                'feature_cols': FEATURE_COLS,
                'seq_length': SEQ_LENGTH,
                'val_acc': val_acc
            }
            # torch.compile 사용 시 state_dict 키 접두사 처리 등 주의 필요하나 
            # 단순 저장엔 문제 없는 경우가 많음. 에러 시 ._orig_mod 사용
            torch.save(save_dict, MODEL_SAVE_PATH)
        else:
            patience_curr += 1
            if patience_curr >= PATIENCE:
                print("🛑 Early Stopping")
                break
                
    # 그래프
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.legend(); plt.savefig('train_boost_result.png')
    print("✅ 완료.")

if __name__ == "__main__":
    # 윈도우 멀티프로세싱 에러 방지
    multiprocessing.freeze_support()
    train()