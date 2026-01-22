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
        self.map_ys = [] # 맵에 존재하는 모든 발판의 Y좌표들
        self.map_min_x = 0
        self.map_max_x = 1366
        self.map_min_y = 0
        self.map_max_y = 768

    def load_map(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.platforms = data.get('platforms', [])
                
                if not self.platforms: 
                    print("⚠️ 경고: 맵 파일에 발판 정보가 없습니다.")
                    return

                # 맵의 범위 계산
                xs = [p['x_start'] for p in self.platforms] + [p['x_end'] for p in self.platforms]
                ys = [p['y'] for p in self.platforms]
                
                # 피처 정규화를 위한 맵 경계 설정
                self.map_min_x = min(xs) - 20
                self.map_max_x = max(xs) + 20
                self.map_min_y = min(ys) - 100
                self.map_max_y = max(ys) + 20
                
                # [핵심] 맵의 유니크한 발판 Y좌표 목록 저장 (보정 기준점)
                self.map_ys = sorted(list(set(ys)))
                
                print(f"🗺️ 맵 로드 완료: 발판 Y좌표 목록 -> {self.map_ys}")
                
        except Exception as e:
            print(f"❌ 맵 로드 실패: {e}")

    def get_features(self, x, y):
        # 맵 밖으로 나가는 거리 등 상대적 거리 계산
        d_up = abs(y - self.map_min_y)
        d_down = abs(self.map_max_y - y)
        d_left = abs(x - self.map_min_x)
        d_right = abs(self.map_max_x - x)
        
        # 가장 가까운 발판과의 거리 계산
        for p in self.platforms:
            if p['x_start'] <= x <= p['x_end']:
                diff_y = p['y'] - y
                if diff_y > 0: d_down = min(d_down, diff_y) # 발판이 아래에 있음
                elif diff_y < 0: d_up = min(d_up, abs(diff_y)) # 발판이 위에 있음
            
            # 발판 좌우 끝점과의 거리
            if abs(p['y'] - y) < 20:
                if p['x_end'] < x: d_left = min(d_left, x - p['x_end'])
                elif p['x_start'] > x: d_right = min(d_right, p['x_start'] - x)

        return pd.Series([
            100/(d_up+1), 100/(d_down+1), 100/(d_left+1), 100/(d_right+1),
            math.sqrt((x-self.map_min_x)**2 + (y-self.map_min_y)**2),
            math.sqrt((x-self.map_max_x)**2 + (y-self.map_min_y)**2),
            math.sqrt((x-self.map_min_x)**2 + (self.map_max_y-y)**2),
            math.sqrt((self.map_max_x-x)**2 + (self.map_max_y-y)**2)
        ])

def fill_action_gaps(df, duration_limit=0.7):
    """행동 간의 빈 공백을 채워줌 (Holding 효과)"""
    timestamps = df['timestamp'].values
    actions = df['key_pressed'].fillna('None').astype(str).values
    filled = []
    last_act = 'None'; last_t = 0.0
    ignore = ['esc', 'f1', 'caps_lock', 'unknown', 'None', 'nan']
    for t, a in zip(timestamps, actions):
        if a not in ignore: last_act = a; last_t = t; filled.append(a)
        elif last_act != 'None' and (t - last_t) <= duration_limit: filled.append(last_act)
        else: filled.append('None')
    df['key_pressed'] = filled
    return df

def detect_double_jumps(df):
    """연속 점프를 더블 점프로 변환"""
    is_jump = df['key_pressed'].str.contains('jump', case=False, na=False)
    jump_indices = df[is_jump].index
    if len(jump_indices) < 2: return df
    
    timestamps = df.loc[jump_indices, 'timestamp'].values
    keys = df.loc[jump_indices, 'key_pressed'].values
    
    for i in range(1, len(jump_indices)):
        dt = timestamps[i] - timestamps[i-1]
        if 0.1 < dt < 0.6: # 0.1~0.6초 사이 연속 입력
            original_key = keys[i]
            if 'double_jump' not in original_key:
                new_key = original_key.replace('jump', 'double_jump')
                df.at[jump_indices[i], 'key_pressed'] = new_key
                
    return df

def upgrade_csv_files():
    root = tk.Tk(); root.withdraw()
    
    print("\nStep 1. 기준이 될 맵 파일(.json)을 선택하세요.")
    map_path = filedialog.askopenfilename(title="맵 JSON 파일 선택", filetypes=[("JSON", "*.json")])
    if not map_path: return
    extractor = AdvancedFeatureExtractor()
    extractor.load_map(map_path)

    if not extractor.map_ys:
        print("❌ 맵 데이터를 불러오지 못했습니다.")
        return

    print("\nStep 2. 학습용 CSV 파일들을 선택하세요.")
    files = filedialog.askopenfilenames(title="CSV 파일 선택", filetypes=[("CSV", "*.csv")])
    if not files: return

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'player_x' not in df.columns or 'player_y' not in df.columns: 
                print(f"⏩ 스킵 (좌표 없음): {os.path.basename(f)}")
                continue
            
            print(f"\n🔄 처리 중: {os.path.basename(f)}")
            
            # =========================================================
            # [핵심] 데이터 주도형 자동 오프셋 보정 (Data-Driven Calibration)
            # =========================================================
            
            # 1. 캐릭터가 가장 빈번하게 있었던 Y좌표 찾기 (바닥일 확률 99%)
            # value_counts().idxmax()는 최빈값(Mode)을 반환합니다.
            player_ground_y = df['player_y'].value_counts().idxmax()
            
            # 2. 맵 파일에서 이와 가장 가까운 발판 Y좌표 찾기
            # min(iterable, key=function)을 사용하여 차이가 가장 작은 값을 찾음
            closest_map_y = min(extractor.map_ys, key=lambda y: abs(y - player_ground_y))
            
            # 3. 보정값(Offset) 계산
            offset = closest_map_y - player_ground_y
            
            print(f"   📊 보정 분석: 캐릭터 바닥({player_ground_y}) vs 맵 발판({closest_map_y})")
            
            if offset != 0:
                df['player_y'] = df['player_y'] + offset
                print(f"   ✅ 오프셋 적용: {offset:+d} px (좌표 동기화 완료)")
            else:
                print(f"   ✨ 보정 불필요: 이미 완벽하게 일치합니다.")
            
            # =========================================================

            # 1. Action Filling
            df = fill_action_gaps(df)
            
            # 2. Double Jump Detection
            df = detect_double_jumps(df)
            
            # 3. Delta & Features
            df['delta_x'] = df['player_x'].diff().fillna(0)
            df['delta_y'] = df['player_y'].diff().fillna(0)
            
            # 피처 추출 (이제 보정된 player_y를 사용하므로 정확함)
            feats = df.apply(lambda row: extractor.get_features(row['player_x'], row['player_y']), axis=1)
            feats.columns = ['inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right', 
                           'corner_tl', 'corner_tr', 'corner_bl', 'corner_br']
            
            final = pd.concat([df, feats], axis=1)
            
            d, n = os.path.split(f)
            save_path = os.path.join(d, f"upgraded_{n}")
            final.to_csv(save_path, index=False)
            print(f"   💾 저장 완료")
            
        except Exception as e: print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    upgrade_csv_files()