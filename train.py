import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_model():
    # 1. 파일 선택
    root = tk.Tk()
    root.withdraw()
    print("📂 학습할 CSV 데이터 파일을 선택하세요...")
    
    # 여러 파일을 한 번에 선택해서 합쳐서 학습할 수 있게 수정
    csv_files = filedialog.askopenfilenames(
        title="학습 데이터 선택 (다중 선택 가능)",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not csv_files:
        print("❌ 파일이 선택되지 않았습니다.")
        return

    # 2. 데이터 병합 및 로드
    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ 파일 로드 오류 ({f}): {e}")
            
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    print(f"📊 총 데이터 개수: {len(df)}개")

    # 3. 데이터 전처리
    df['key_pressed'] = df['key_pressed'].fillna('None')
    
    # [중요] 플랫폼 ID가 -1(허공)인 경우도 하나의 상태로 학습
    df['platform_id'] = df['platform_id'].fillna(-1)

    # 사용할 특성 정의 (platform_id 추가됨!)
    feature_cols = ['player_x', 'player_y', 'entropy', 'platform_id']
    target_col = 'key_pressed'

    # 필요한 컬럼 체크
    for col in feature_cols:
        if col not in df.columns:
            print(f"❌ 데이터에 '{col}' 컬럼이 없습니다. (구버전 데이터일 수 있음)")
            return

    X = df[feature_cols]
    y = df[target_col]

    # 4. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 모델 학습 (설정 강화)
    print("⏳ 모델 학습 중...")
    model = RandomForestClassifier(
        n_estimators=200,           # 나무 개수 증가
        max_depth=20,               # 과적합 방지
        class_weight='balanced',    # [핵심] 데이터가 적은 행동도 중요하게 취급
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 6. 성능 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 모델 정확도: {acc:.4f}")
    print("\n분류 리포트:\n", classification_report(y_test, y_pred, zero_division=0))

    # 7. 모델 저장
    if acc < 0.5:
        print("⚠️ 경고: 정확도가 너무 낮습니다 (50% 미만). 더 많은 데이터를 수집하세요.")
        
    save_path = filedialog.asksaveasfilename(
        title="모델 저장",
        defaultextension=".pkl",
        filetypes=[("Pickle files", "*.pkl")],
        initialfile="kinesis_hunt_model_v2.pkl"
    )
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"💾 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    train_model()