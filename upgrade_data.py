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
        # 맵의 경계(Boundaries) 초기화 (기본값)
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
                
                # 맵 끝부분에 약간의 여유(Margin)를 둠
                self.map_min_x = min(xs) - 20
                self.map_max_x = max(xs) + 20
                self.map_min_y = min(ys) - 100 # 위쪽은 점프 높이 고려
                self.map_max_y = max(ys) + 20
                
                print(f"🗺️ 맵 로드 완료: {len(self.platforms)}개 발판")
                print(f"   범위: X({self.map_min_x}~{self.map_max_x}), Y({self.map_min_y}~{self.map_max_y})")
                
        except Exception as e:
            print(f"❌ 맵 로드 실패: {e}")

    def get_features(self, x, y):
        """좌표(x, y)를 받아 8개의 고급 특성 반환"""
        
        # 1. 물리적 거리 계산 (초기값: 맵 끝 벽까지의 거리)
        d_up = abs(y - self.map_min_y)
        d_down = abs(self.map_max_y - y)
        d_left = abs(x - self.map_min_x)
        d_right = abs(self.map_max_x - x)
        
        # 발판과의 거리 비교 (더 가까운 장애물이 있으면 업데이트)
        for p in self.platforms:
            # X축이 겹칠 때 (수직 거리)
            if p['x_start'] <= x <= p['x_end']:
                diff_y = p['y'] - y
                if diff_y > 0: # 내 발 밑에 발판이 있음 (Down 거리)
                    d_down = min(d_down, diff_y)
                elif diff_y < 0: # 내 머리 위에 발판이 있음 (Up 거리)
                    d_up = min(d_up, abs(diff_y))
            
            # Y축이 비슷할 때 (수평 거리, 오차범위 20px)
            if abs(p['y'] - y) < 20:
                if p['x_end'] < x: # 내 왼쪽에 발판 끝이 있음
                    d_left = min(d_left, x - p['x_end'])
                elif p['x_start'] > x: # 내 오른쪽에 발판 시작이 있음
                    d_right = min(d_right, p['x_start'] - x)

        # 2. [핵심] 위기 감지 센서 (거리 역수 변환)
        # 거리가 0에 가까울수록 값이 100에 가깝게 폭증함
        # 수식: 100 / (거리 + 1)
        inv_up = 100 / (d_up + 1)
        inv_down = 100 / (d_down + 1)
        inv_left = 100 / (d_left + 1)
        inv_right = 100 / (d_right + 1)

        # 3. 네비게이션 센서 (모서리까지의 직선 거리)
        # 맵의 절대적인 위치를 파악하는 데 도움
        corner_tl = math.sqrt((x - self.map_min_x)**2 + (y - self.map_min_y)**2) # 좌상
        corner_tr = math.sqrt((x - self.map_max_x)**2 + (y - self.map_min_y)**2) # 우상
        corner_bl = math.sqrt((x - self.map_min_x)**2 + (self.map_max_y - y)**2) # 좌하 (Y축 주의)
        corner_br = math.sqrt((self.map_max_x - x)**2 + (self.map_max_y - y)**2) # 우하

        return pd.Series([
            inv_up, inv_down, inv_left, inv_right,
            corner_tl, corner_tr, corner_bl, corner_br
        ])

def upgrade_csv_files():
    # GUI 창 숨기기
    root = tk.Tk()
    root.withdraw()
    
    # 1. 맵 파일 선택 (JSON)
    print("Step 1. 맵 데이터 파일(Rocky_Overlook3.json 등)을 선택하세요...")
    map_path = filedialog.askopenfilename(
        title="맵 JSON 선택",
        filetypes=[("JSON files", "*.json")]
    )
    if not map_path:
        print("❌ 맵 파일이 선택되지 않았습니다.")
        return

    extractor = AdvancedFeatureExtractor()
    extractor.load_map(map_path)

    # 2. CSV 파일들 선택 (다중 선택 가능)
    print("Step 2. 변환할 CSV 파일들을 선택하세요 (여러 개 선택 가능)...")
    csv_files = filedialog.askopenfilenames(
        title="CSV 데이터 선택",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not csv_files:
        print("❌ CSV 파일이 선택되지 않았습니다.")
        return

    print(f"\n📊 총 {len(csv_files)}개의 파일을 변환합니다...")

    for file_path in csv_files:
        try:
            # 파일 읽기
            df = pd.read_csv(file_path)
            
            # 필수 컬럼 확인
            if 'player_x' not in df.columns or 'player_y' not in df.columns:
                print(f"⚠️ 스킵 (좌표 정보 없음): {os.path.basename(file_path)}")
                continue
            
            print(f"🔄 처리 중: {os.path.basename(file_path)} ...")
            
            # 특성 계산 적용
            new_features = df.apply(
                lambda row: extractor.get_features(row['player_x'], row['player_y']), 
                axis=1
            )
            
            # 컬럼명 지정
            new_features.columns = [
                'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right',
                'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
            ]
            
            # 기존 데이터와 병합
            df_final = pd.concat([df, new_features], axis=1)
            
            # 파일 저장 (파일명 앞에 'upgraded_' 붙임)
            dir_name, base_name = os.path.split(file_path)
            save_path = os.path.join(dir_name, f"upgraded_{base_name}")
            
            df_final.to_csv(save_path, index=False)
            print(f"✅ 저장 완료: {save_path}")

        except Exception as e:
            print(f"❌ 에러 발생 ({os.path.basename(file_path)}): {e}")

    print("\n✨ 모든 작업이 완료되었습니다! 'upgraded_...' 파일들을 학습에 사용하세요.")

if __name__ == "__main__":
    upgrade_csv_files()