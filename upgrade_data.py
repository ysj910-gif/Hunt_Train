import pandas as pd
import json
import os
import math
import tkinter as tk
from tkinter import filedialog
import numpy as np

class AdvancedFeatureExtractor:
    def __init__(self):
        self.platforms = []
        # 맵 경계 기본값
        self.map_min_x = 0
        self.map_max_x = 1366
        self.map_min_y = 0
        self.map_max_y = 768

    def load_map(self, file_path):
        """맵 JSON 파일 로드 및 경계 계산"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.platforms = data.get('platforms', [])
                
                if not self.platforms:
                    print("⚠️ 발판 정보가 없습니다. 기본 해상도(1366x768)를 기준으로 합니다.")
                    return

                # 맵의 실제 크기(Bounding Box) 자동 계산
                xs = [p['x_start'] for p in self.platforms] + [p['x_end'] for p in self.platforms]
                ys = [p['y'] for p in self.platforms]
                
                self.map_min_x = min(xs) - 20
                self.map_max_x = max(xs) + 20
                self.map_min_y = min(ys) - 100 
                self.map_max_y = max(ys) + 20
                
                print(f"🗺️ 맵 로드 완료: {len(self.platforms)}개 발판")
                print(f"   범위: X({self.map_min_x}~{self.map_max_x}), Y({self.map_min_y}~{self.map_max_y})")
                
        except Exception as e:
            print(f"❌ 맵 로드 실패: {e}")

    def get_features(self, x, y):
        """좌표(x, y)를 받아 8개의 고급 특성 반환"""
        d_up = abs(y - self.map_min_y)
        d_down = abs(self.map_max_y - y)
        d_left = abs(x - self.map_min_x)
        d_right = abs(self.map_max_x - x)
        
        # 발판과의 거리 비교 (가장 가까운 장애물 찾기)
        for p in self.platforms:
            if p['x_start'] <= x <= p['x_end']:
                diff_y = p['y'] - y
                if diff_y > 0: d_down = min(d_down, diff_y)
                elif diff_y < 0: d_up = min(d_up, abs(diff_y))
            
            if abs(p['y'] - y) < 20:
                if p['x_end'] < x: d_left = min(d_left, x - p['x_end'])
                elif p['x_start'] > x: d_right = min(d_right, p['x_start'] - x)

        # 위기 감지 센서 (거리 역수)
        inv_up = 100 / (d_up + 1)
        inv_down = 100 / (d_down + 1)
        inv_left = 100 / (d_left + 1)
        inv_right = 100 / (d_right + 1)

        # 네비게이션 센서 (모서리 거리)
        corner_tl = math.sqrt((x - self.map_min_x)**2 + (y - self.map_min_y)**2)
        corner_tr = math.sqrt((x - self.map_max_x)**2 + (y - self.map_min_y)**2)
        corner_bl = math.sqrt((x - self.map_min_x)**2 + (self.map_max_y - y)**2)
        corner_br = math.sqrt((self.map_max_x - x)**2 + (self.map_max_y - y)**2)

        return pd.Series([
            inv_up, inv_down, inv_left, inv_right,
            corner_tl, corner_tr, corner_bl, corner_br
        ])

def fill_action_gaps(df, duration_limit=0.7):
    """
    [핵심] 키 입력 사이의 공백(None)을 이전 행동으로 채워줌 (Wait 문제 해결)
    duration_limit: 최대 몇 초까지 행동을 유지할지 (기본 0.7초)
    """
    timestamps = df['timestamp'].values
    actions = df['key_pressed'].fillna('None').astype(str).values
    
    filled_actions = []
    last_action = 'None'
    last_time = 0.0
    
    # 무시할 시스템 키 (이런 키는 연장하지 않음)
    ignore_keys = ['esc', 'f1', 'caps_lock', 'unknown', 'None', 'nan']

    for t, a in zip(timestamps, actions):
        # 새로운 유효한 행동이 나오면 갱신
        if a not in ignore_keys:
            last_action = a
            last_time = t
            filled_actions.append(a)
        else:
            # 행동이 없는 경우 (None)
            # 마지막 행동이 유효하고, 시간이 duration_limit 이내라면 -> 행동 연장
            if last_action != 'None' and (t - last_time) <= duration_limit:
                filled_actions.append(last_action)
            else:
                filled_actions.append('None') # 시간 초과 시 진짜 Idle
                
    df['key_pressed'] = filled_actions
    return df

def upgrade_csv_files():
    root = tk.Tk()
    root.withdraw()
    
    print("Step 1. 맵 데이터 파일(.json)을 선택하세요...")
    map_path = filedialog.askopenfilename(title="맵 JSON 선택", filetypes=[("JSON files", "*.json")])
    if not map_path: return

    extractor = AdvancedFeatureExtractor()
    extractor.load_map(map_path)

    print("Step 2. 변환할 CSV 파일들을 선택하세요 (여러 개 가능)...")
    csv_files = filedialog.askopenfilenames(title="CSV 데이터 선택", filetypes=[("CSV files", "*.csv")])
    if not csv_files: return

    print(f"\n📊 총 {len(csv_files)}개의 파일을 변환합니다...")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            if 'player_x' not in df.columns:
                print(f"⚠️ 스킵 (좌표 정보 없음): {os.path.basename(file_path)}")
                continue
            
            print(f"🔄 처리 중: {os.path.basename(file_path)} ...")
            
            # 1. 액션 지속시간 보정 (Action Filling)
            df = fill_action_gaps(df, duration_limit=0.7)
            
            # 2. 특성 계산 적용
            new_features = df.apply(
                lambda row: extractor.get_features(row['player_x'], row['player_y']), 
                axis=1
            )
            
            new_features.columns = [
                'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right',
                'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
            ]
            
            df_final = pd.concat([df, new_features], axis=1)
            
            dir_name, base_name = os.path.split(file_path)
            save_path = os.path.join(dir_name, f"upgraded_{base_name}")
            
            df_final.to_csv(save_path, index=False)
            print(f"✅ 저장 완료: {save_path}")

        except Exception as e:
            print(f"❌ 에러 발생 ({os.path.basename(file_path)}): {e}")

    print("\n✨ 모든 작업이 완료되었습니다! 변환된 파일들로 다시 학습(train_lstm.py)해주세요.")

if __name__ == "__main__":
    upgrade_csv_files()