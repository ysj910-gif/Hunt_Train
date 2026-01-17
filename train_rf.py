import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_rf_model():
    # 1. 파일 선택
    root = tk.Tk()
    root.withdraw()
    print("🌲 [RandomForest] 학습할 CSV 데이터 파일을 선택하세요...")
    
    csv_files = filedialog.askopenfilenames(
        title="학습 데이터 선택 (다중 선택 가능)",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not csv_files:
        print("❌ 파일이 선택되지 않았습니다.")
        return

    # 2. 데이터 로드 및 병합
    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            # 필수 컬럼 확인
            if 'player_x' in temp_df.columns:
                df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ 로드 실패 ({f}): {e}")
            
    if not df_list:
        print("❌ 학습할 데이터가 없습니다.")
        return
        
    df = pd.concat(df_list, ignore_index=True)
    print(f"📊 원본 데이터: {len(df)}개")

    # 3. 데이터 전처리 (핵심: 멍때리기 및 노이즈 제거)
    
    # (1) 결측치 처리
    df['key_pressed'] = df['key_pressed'].fillna('None')
    if 'platform_id' not in df.columns:
        df['platform_id'] = -1
    df['platform_id'] = df['platform_id'].fillna(-1)

    # (2) 노이즈 키 제거
    ignore_keys = ['media_volume_up', 'esc', 'f1', 'alt_l', 'caps_lock', 'shift']
    df = df[~df['key_pressed'].isin(ignore_keys)]

    # (3) [중요] 'None' 및 'down' 제거 (공격성 강화)
    # 웜업 단계에서도 멍때리지 않게 None을 90% 이상 줄이거나 제거
    print("🧹 데이터 정제 중... (None, down 제거)")
    df = df[df['key_pressed'] != 'None'] # None 완전 제거 (즉시 반응 유도)
    df = df[df['key_pressed'] != 'down'] # 광클 유발하는 앉기 제거

    print(f"✨ 정제된 데이터: {len(df)}개 (학습 준비 완료)")

    if len(df) < 100:
        print("⚠️ 데이터가 너무 적습니다. 녹화를 더 해주세요.")
        return

    # 4. 특성(Feature) 및 정답(Target) 설정
    # LSTM과 달리 RF는 시간 흐름(Context)을 모르므로 
    # 현재 위치와 상태만 보고 즉각 반응하도록 합니다.
    feature_cols = ['player_x', 'player_y', 'entropy', 'platform_id']
    target_col = 'key_pressed'
    
    # 쿨타임 정보(ult_ready 등)가 CSV에 있다면 추가 활용 (없으면 4개만 사용)
    if 'ult_ready' in df.columns and 'sub_ready' in df.columns:
        feature_cols.extend(['ult_ready', 'sub_ready'])
        print(f"💡 고급 특성 포함 학습: {feature_cols}")

    X = df[feature_cols]
    y = df[target_col]

    # 5. 학습 및 평가
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⏳ 모델 학습 중... (Random Forest)")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,          # 너무 깊지 않게 (일반화)
        min_samples_leaf=2,    # 노이즈 과적합 방지
        class_weight='balanced', 
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 정확도 출력
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 모델 정확도: {acc:.2%}")
    # print(classification_report(y_test, y_pred, zero_division=0))

    # 6. 모델 저장
    save_path = filedialog.asksaveasfilename(
        title="RF 모델 저장 (.pkl)",
        defaultextension=".pkl",
        filetypes=[("Pickle files", "*.pkl")],
        initialfile="rf_warmup_model.pkl"
    )
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"💾 저장 완료: {save_path}")
        print("👉 이제 GUI에서 'Load RF' 버튼으로 이 파일을 불러오세요!")

if __name__ == "__main__":
    train_rf_model()