#train\train_physics.py
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
import multiprocessing

# =============================================================================
# [핵심] 상위 폴더(프로젝트 루트) 연결
# train 폴더 안에 있어도 바깥에 있는 'platform_manager.py'를 찾을 수 있게 해줍니다.
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# =============================================================================
# [1] 모델 클래스 직접 정의 (rune_solver.py 참조 안 함 -> 오류 해결)
# =============================================================================
class HybridPhysicsNet(nn.Module):
    def __init__(self, num_actions):
        super(HybridPhysicsNet, self).__init__()
        # 물리 파라미터 (속도 X, 속도 Y, 중력 계수)
        self.physics_params = nn.Embedding(num_actions, 3)
        self.physics_params.weight.data.uniform_(0.1, 1.0)
        
        # 행동 임베딩 (잔차 학습용)
        self.action_emb = nn.Embedding(num_actions, 8)
        
        # 잔차 신경망 (물리 공식으로 설명 안 되는 미세 움직임 보정)
        self.residual_net = nn.Sequential(
            nn.Linear(8 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, action_idx, is_grounded):
        if is_grounded.dim() > 1:
            is_grounded = is_grounded.squeeze(1)

        # 1. 기본 물리 연산 (F = ma 기반 추정)
        params = self.physics_params(action_idx)
        phys_vx = params[:, 0] * 10.0
        phys_vy = params[:, 1] * 10.0
        gravity = params[:, 2] * 5.0 * (1.0 - is_grounded) # 공중에 있을 때만 중력 적용
        
        base_dx = phys_vx
        base_dy = phys_vy + gravity
        base_move = torch.stack([base_dx, base_dy], dim=1)
        
        # 2. 잔차 보정 (Residual Learning)
        emb = self.action_emb(action_idx)
        cat_ground = is_grounded.unsqueeze(1)
        cat = torch.cat([emb, cat_ground], dim=1)
        residual = self.residual_net(cat)
        
        return base_move + residual

# =============================================================================
# [2] 플랫폼 매니저 로드 (없어도 작동하도록 예외 처리)
# =============================================================================
try:
    from platform_manager import PlatformManager
    print("✅ 플랫폼 매니저(PlatformManager) 로드 성공")
except ImportError:
    print("⚠️ 'platform_manager.py'를 상위 폴더에서 찾을 수 없습니다. (지형 인식 정확도 감소)")
    PlatformManager = None

# =============================================================================
# [3] 학습 설정 (RTX 4070 Super 최적화)
# =============================================================================
EPOCHS = 500          
BATCH_SIZE = 4096     
LEARNING_RATE = 0.01  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"\n🚀 물리 엔진 독립 학습 시작 (Device: {DEVICE})")
    
    root = tk.Tk(); root.withdraw()

    # [Step 1] 맵 데이터 로드
    pm = None
    if PlatformManager:
        print("\nStep 1. 맵 파일(.json)을 선택하세요... (선택 취소 시 단순 모드)")
        map_path = filedialog.askopenfilename(
            initialdir=parent_dir, # 파일 선택 창을 상위 폴더에서 시작
            title="맵 JSON 선택", 
            filetypes=[("JSON files", "*.json")]
        )
        if map_path:
            pm = PlatformManager()
            pm.load_platforms(map_path)
            print(f"   맵 로드 완료: {os.path.basename(map_path)}")

    # [Step 2] 데이터 파일 선택
    print("\nStep 2. 학습할 CSV 데이터 파일들을 선택하세요 (upgraded_...csv 권장)...")
    csv_files = filedialog.askopenfilenames(
        initialdir=parent_dir,
        title="학습 데이터 선택", 
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_files:
        print("❌ 파일이 선택되지 않았습니다."); return

    # [Step 3] 데이터 로드 및 정제
    print(f"⏳ {len(csv_files)}개 파일 분석 중...")
    
    actions_list = []
    states_list = []
    movements_list = []
    
    total_rows = 0
    valid_rows = 0
    skipped_static = 0
    IGNORE_KEYS = ['media_volume_up', 'esc', 'f1', 'caps_lock', 'unknown', 'alt_l', 'shift', 'ctrl', 'tab', 'enter']

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

                if any(ig in key.lower() for ig in IGNORE_KEYS): continue

                # 정지 데이터 필터링
                if abs(dx) < 1.0 and abs(dy) < 1.0:
                    if 'jump' not in key.lower() and np.random.rand() > 0.1: 
                        skipped_static += 1
                        continue

                # 지상 판정
                is_grounded = 0.0
                if pm:
                    if pm.get_current_platform(xs[i], ys[i]) != -1: is_grounded = 1.0
                else:
                    if abs(dy) < 2.0: is_grounded = 1.0

                actions_list.append(key)
                states_list.append(is_grounded)
                movements_list.append([dx, dy])
                valid_rows += 1
                    
        except Exception as e:
            print(f"❌ 에러: {e}")

    print(f"📊 정제 결과: {valid_rows}행 학습 (제자리 {skipped_static}행 삭제됨)")
    if valid_rows == 0: return

    # [Step 4] 텐서 변환
    encoder = LabelEncoder()
    action_ids = encoder.fit_transform(actions_list)
    num_actions = len(encoder.classes_)
    print(f"🏷️ 학습할 행동 클래스: {num_actions}개")
    
    X_actions = torch.LongTensor(action_ids).to(DEVICE)
    X_states = torch.FloatTensor(states_list).unsqueeze(1).to(DEVICE)
    y_vectors = torch.FloatTensor(movements_list).to(DEVICE)
    
    dataset = TensorDataset(X_actions, X_states, y_vectors)
    
    # 4070 Super 최적화 (num_workers)
    num_workers = min(4, multiprocessing.cpu_count()) 
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    
    # [Step 5] 모델 학습
    model = HybridPhysicsNet(num_actions).to(DEVICE)
    try: model = torch.compile(model) 
    except: pass

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss() 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    print("\n🔥 정밀 학습 시작...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for b_act, b_state, b_target in dataloader:
            b_act = b_act.to(DEVICE, non_blocking=True)
            b_state = b_state.to(DEVICE, non_blocking=True)
            b_target = b_target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            pred = model(b_act, b_state)
            loss = criterion(pred, b_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    # [Step 6] 저장 (루트 폴더에 저장)
    # **중요**: main.py가 있는 상위 폴더에 저장해야 봇이 바로 읽습니다.
    save_path = os.path.join(parent_dir, "physics_hybrid_model.pth")
    
    # torch.compile 사용 시 원본 state_dict 저장
    state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    
    torch.save({
        'model_state': state_dict, 
        'encoder': encoder,
        'input_size': num_actions
    }, save_path)
    print(f"\n💾 저장 완료: {save_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()