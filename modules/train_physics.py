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

# 1. 모듈 불러오기 (경로 예외 처리)
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

# 2. 설정
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"\n🚀 물리 엔진 학습 시작 (Device: {DEVICE})")
    
    # [Step 1] 맵 데이터 로드
    root = tk.Tk(); root.withdraw()
    pm = None
    if PlatformManager:
        print("\nStep 1. 맵 파일(.json)을 선택하세요...")
        map_path = filedialog.askopenfilename(title="맵 JSON 선택", filetypes=[("JSON", "*.json")])
        if map_path:
            pm = PlatformManager()
            pm.load_platforms(map_path)
        else:
            print("ℹ️ 맵 파일 선택 안 함 (공중/지상 판정을 단순화합니다)")

    # [Step 2] 데이터 파일 선택 (여기가 요청하신 기능!)
    print("\nStep 2. 학습할 CSV 데이터 파일들을 선택하세요 (upgraded_...csv 권장)...")
    csv_files = filedialog.askopenfilenames(
        title="학습 데이터 선택", 
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not csv_files:
        print("❌ 파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    # [Step 3] 데이터 로드 및 전처리
    print(f"⏳ {len(csv_files)}개 파일 분석 중...")
    
    actions_list = []
    states_list = []
    movements_list = []
    
    total_rows = 0
    valid_rows = 0

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # 필수 컬럼 확인
            cols = ['timestamp', 'player_x', 'player_y', 'key_pressed']
            if not all(c in df.columns for c in cols): 
                print(f"⚠️ 스킵 (필수 컬럼 부족): {os.path.basename(file)}")
                continue
            
            total_rows += len(df)
            
            # [핵심 수정] 빈 키 입력(NaN)을 'None'으로 채우기 (에러 방지)
            df['key_pressed'] = df['key_pressed'].fillna('None')
            
            # 데이터 순회하며 물리량(변화량) 추출
            # (Vectorized 연산 대신 루프를 돌며 정밀하게 체크)
            times = df['timestamp'].values
            xs = df['player_x'].values
            ys = df['player_y'].values
            keys = df['key_pressed'].values
            
            for i in range(len(df) - 1):
                dt = times[i+1] - times[i]
                
                # 프레임이 끊기지 않은 경우만 학습 (0.01초 ~ 0.5초 사이)
                if 0.01 <= dt <= 0.5:
                    dx = xs[i+1] - xs[i]
                    dy = ys[i+1] - ys[i]
                    key = str(keys[i]) # 확실하게 문자열로 변환
                    
                    # 지상/공중 판정
                    is_grounded = 0.0
                    if pm:
                        # PlatformManager가 있으면 정밀 판정
                        if pm.get_current_platform(xs[i], ys[i]):
                            is_grounded = 1.0
                    else:
                        # 없으면 대충 Y 변화량이 적을 때 땅이라고 가정
                        if abs(dy) < 1.0: is_grounded = 1.0

                    actions_list.append(key)
                    states_list.append(is_grounded)
                    movements_list.append([dx, dy])
                    valid_rows += 1
                    
        except Exception as e:
            print(f"❌ 파일 읽기 에러 ({os.path.basename(file)}): {e}")

    print(f"📊 데이터 로드 완료: 총 {total_rows}행 중 {valid_rows}행 유효")
    
    if valid_rows == 0:
        print("❌ 학습할 유효 데이터가 없습니다.")
        return

    # [Step 4] 텐서 변환
    encoder = LabelEncoder()
    action_ids = encoder.fit_transform(actions_list)
    num_actions = len(encoder.classes_)
    print(f"🏷️ 학습할 행동 종류: {num_actions}개 ({encoder.classes_})")
    
    X_actions = torch.LongTensor(action_ids).to(DEVICE)
    X_states = torch.FloatTensor(states_list).unsqueeze(1).to(DEVICE) # [N, 1]
    y_vectors = torch.FloatTensor(movements_list).to(DEVICE)          # [N, 2]
    
    dataset = TensorDataset(X_actions, X_states, y_vectors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # [Step 5] 모델 학습
    model = HybridPhysicsNet(num_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # 평균 제곱 오차

    print("\n🔥 학습 시작...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for b_act, b_state, b_target in dataloader:
            optimizer.zero_grad()
            
            # 모델 예측
            pred = model(b_act, b_state)
            
            # 오차 계산
            loss = criterion(pred, b_target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss(MSE): {avg_loss:.4f}")

    # [Step 6] 모델 저장
    save_path = "physics_hybrid_model.pth"
    save_dict = {
        'model_state': model.state_dict(),
        'encoder': encoder
    }
    torch.save(save_dict, save_path)
    print(f"\n💾 물리 엔진 저장 완료: {save_path}")
    print("   이제 gui.py를 실행하면 룬을 찾을 때 이 물리 엔진을 사용합니다!")

if __name__ == "__main__":
    main()