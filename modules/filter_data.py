# filter_data.py
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

def filter_csv():
    # 1. 파일 선택
    root = tk.Tk(); root.withdraw()
    print("🧹 필터링할 봇 플레이 데이터(CSV)를 선택하세요...")
    files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if not files: return

    total_rows = 0
    saved_rows = 0

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            if 'kill_count' not in df.columns or 'timestamp' not in df.columns:
                print(f"⚠️ 스킵 (형식 불일치): {os.path.basename(file_path)}")
                continue

            original_len = len(df)
            total_rows += original_len

            # --- [필터링 로직] ---
            
            # 1. 킬 카운트가 증가한 시점 찾기
            # kill_count가 이전 행보다 커진 순간(몬스터 처치)을 True로 표시
            df['kill_diff'] = df['kill_count'].diff().fillna(0)
            kill_moments = df.index[df['kill_diff'] > 0].tolist()

            # 2. 유효 구간 설정 (몬스터 처치 전 2초 ~ 처치 후 0.5초)
            # 공격 행동과 그에 따른 이동만 학습하기 위함
            valid_indices = set()
            fps = 30 # 대략적인 초당 프레임 수 (loop 속도 0.033 기준)
            window_before = 2 * fps # 2초 전
            window_after = 0.5 * fps # 0.5초 후

            for idx in kill_moments:
                start = max(0, int(idx - window_before))
                end = min(len(df), int(idx + window_after))
                valid_indices.update(range(start, end))

            # 3. 이동 성공 데이터 추가 (선택 사항)
            # 제자리(벽비비기)가 아닌 경우만 포함하려면 좌표 변화량 체크 로직 추가 가능
            # 여기서는 간단하게 킬 관련 데이터만 남김

            # 필터링 적용
            filtered_df = df.iloc[sorted(list(valid_indices))]
            
            # ---------------------

            if len(filtered_df) > 0:
                # 파일 저장 (파일명 앞에 'filtered_' 붙임)
                dir_name, base_name = os.path.split(file_path)
                save_name = os.path.join(dir_name, f"filtered_{base_name}")
                filtered_df.to_csv(save_name, index=False)
                print(f"✅ 저장 완료: {base_name} ({original_len} -> {len(filtered_df)}행)")
                saved_rows += len(filtered_df)
            else:
                print(f"🗑️ 모두 삭제됨 (유효 행동 없음): {os.path.basename(file_path)}")

        except Exception as e:
            print(f"❌ 에러 발생: {e}")

    print(f"\n📊 요약: 총 {total_rows}행 중 {saved_rows}행({saved_rows/total_rows*100:.1f}%)이 유효 데이터로 선별되었습니다.")

if __name__ == "__main__":
    filter_csv()