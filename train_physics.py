import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
import sys

# 1. 모듈 불러오기
try:
    from modules.rune_solver import HybridPhysicsNet
    print("✅ 모델 클래스(HybridPhysicsNet) 로드 성공")
except ImportError as e:
    print(f"❌ [오류] 'modules/rune_solver.py'를 찾을 수 없습니다: {e}")
    sys.exit(1)

try:
    from platform_manager import PlatformManager
    print("✅ 플랫폼 매니저(PlatformManager) 로드 성공")
except ImportError:
    print("⚠️ 'platform_manager.py'가 없습니다. 지형 인식(땅/공중) 정확도가 떨어질 수 있습니다.")
    PlatformManager = None

# 2. 설정 (하이퍼파라미터 튜닝)
EPOCHS = 150         # 학습 횟수 증가
BATCH_SIZE = 64
LEARNING_RATE = 0.01 # 초기 학습률을 좀 더 높게 설정 (스케줄러가 깎을 것임)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"\n🚀 물리 엔진 정밀 학습 시작 (Device: {DEVICE})")
    
    root = tk.Tk(); root.withdraw()

    # [Step 1] 맵 데이터 로드
    pm = None
    if PlatformManager:
        print("\nStep 1. 맵 파일(.json)을 선택하세요...")
        map_path = filedialog.askopenfilename(title="맵 JSON 선택", filetypes=[("JSON files", "*.json")])
        if map_path:
            pm = PlatformManager()
            pm.load_platforms(map_path)
            print(f"   맵 로드 완료: {os.path.basename(map_path)}")

    # [Step 2] 데이터 파일 선택
    print("\nStep 2. 학습할 CSV 데이터 파일들을 선택하세요 (upgraded_...csv 권장)...")
    csv_files = filedialog.askopenfilenames(title="학습 데이터 선택", filetypes=[("CSV files", "*.csv")])
    if not csv_files:
        print("❌ 파일이 선택되지 않았습니다."); return

    # [Step 3] 데이터 로드 및 정제 (Cleaning)
    print(f"⏳ {len(csv_files)}개 파일 분석 및 정제 중...")
    
    actions_list = []
    states_list = []
    movements_list = []
    
    total_rows = 0
    valid_rows = 0
    skipped_static = 0

    # 무시할 키 목록 (노이즈 제거)
    IGNORE_KEYS = ['media_volume_up', 'esc', 'f1', 'caps_lock', 'unknown', 'alt_l', 'shift', 'ctrl']

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            cols = ['timestamp', 'player_x', 'player_y', 'key_pressed']
            if not all(c in df.columns for c in cols): continue
            
            total_rows += len(df)
            df['key_pressed'] = df['key_pressed'].fillna('None')

            times = df['timestamp'].values
            xs = df['player_x'].values
            ys = df['player_y'].values
            keys = df['key_pressed'].values.astype(str)
            
            for i in range(len(df) - 1):
                dt = times[i+1] - times[i]
                if not (0.01 <= dt <= 0.5): continue

                dx = xs[i+1] - xs[i]
                dy = ys[i+1] - ys[i]
                key = keys[i]

                # 1. 쓸모없는 키 제외
                if any(ig in key.lower() for ig in IGNORE_KEYS):
                    continue

                # 2. [핵심] 정지 데이터(제자리) 과감하게 줄이기 (Under-sampling)
                # 움직임이 거의 없는데(dx, dy < 1) 키도 안 눌렀거나(None) 단순 대기 중이면 90% 확률로 버림
                if abs(dx) < 1.0 and abs(dy) < 1.0:
                    if np.random.rand() > 0.1: # 10%만 남기고 버림
                        skipped_static += 1
                        continue

                # 지상 판정
                is_grounded = 0.0
                if pm:
                    if pm.get_current_platform(xs[i], ys[i]): is_grounded = 1.0
                else:
                    if abs(dy) < 1.0: is_grounded = 1.0

                actions_list.append(key)
                states_list.append(is_grounded)
                movements_list.append([dx, dy])
                valid_rows += 1
                    
        except Exception as e:
            print(f"❌ 에러 ({os.path.basename(file)}): {e}")

    print(f"📊 정제 결과: 원본 {total_rows}행 -> 학습 {valid_rows}행 (제자리 {skipped_static}행 삭제됨)")
    
    if valid_rows == 0: print("❌ 학습할 데이터가 없습니다."); return

    # [Step 4] 텐서 변환
    encoder = LabelEncoder()
    action_ids = encoder.fit_transform(actions_list)
    num_actions = len(encoder.classes_)
    print(f"🏷️ 학습할 행동: {num_actions}개 ({encoder.classes_})")
    
    X_actions = torch.LongTensor(action_ids).to(DEVICE)
    X_states = torch.FloatTensor(states_list).unsqueeze(1).to(DEVICE)
    y_vectors = torch.FloatTensor(movements_list).to(DEVICE)
    
    dataset = TensorDataset(X_actions, X_states, y_vectors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # [Step 5] 모델 학습 (Scheduler & SmoothL1Loss 적용)
    model = HybridPhysicsNet(num_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # [변경] MSE보다 안정적인 SmoothL1Loss 사용
    criterion = nn.SmoothL1Loss() 
    
    # [추가] 학습이 정체되면 LR을 깎는 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    print("\n🔥 정밀 학습 시작...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for b_act, b_state, b_target in dataloader:
            optimizer.zero_grad()
            pred = model(b_act, b_state)
            loss = criterion(pred, b_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # 스케줄러에게 보고
        scheduler.step(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # [Step 6] 저장
    save_path = "physics_hybrid_model.pth"
    torch.save({'model_state': model.state_dict(), 'encoder': encoder}, save_path)
    print(f"\n💾 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    main()