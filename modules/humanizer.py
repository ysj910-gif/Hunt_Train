# modules/humanizer.py
import pandas as pd
import numpy as np
import random
import os

class Humanizer:
    def __init__(self):
        # 1. Personal Model (나의 습관 - 로그에서 학습)
        self.p_mean = 0.05   # 기본값 50ms
        self.p_std = 0.01    # 기본값 10ms
        self.is_fitted = False

        # 2. General Model (일반적인 사람 - Ex-Gaussian 파라미터)
        # 보통 키 누름은 최소 인지 시간 + 처리 시간(Tail)으로 구성됨
        self.g_mu = 0.04     # 일반인 기초 평균 (40ms)
        self.g_sigma = 0.01  # 일반인 기초 편차 (10ms)
        self.g_tau = 0.03    # 지수분포 꼬리 (30ms) -> 가끔 길게 누르는 특징 반영

        # 3. Blending (섞기)
        # 0.0: 완전 일반인 모드 (내 특징 숨김)
        # 1.0: 완전 내 모드
        # 0.3~0.5 추천 (내 습관을 30~50%만 반영하고 나머지는 일반인처럼)
        self.blending_ratio = 0.4 

    def fit_from_logs(self, data_folder="data"):
        """로그 파일에서 사용자 고유의 키 입력 패턴(Fingerprint)을 추출"""
        print("🕵️ Humanizer: 사용자 패턴 분석 및 익명화 준비 중...")
        
        csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
        if not csv_files:
            print("⚠️ 로그 파일이 없어 기본값(일반인 모델)을 사용합니다.")
            return

        durations = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if 'timestamp' not in df.columns: continue
                
                # 키 입력 시간차 계산 (diff)
                # 단순화: 로그의 시간 간격을 키 누름 시간으로 추정 (정교한 분석 필요 시 수정 가능)
                # 여기서는 시뮬레이션을 위해 0.03~0.15초 사이의 값만 유효한 '누름'으로 간주
                
                # 실제로는 press/release 로그가 쌍으로 있어야 정확하지만, 
                # 현재 구조상 step 간격 등을 통해 간접 추정하거나,
                # 만약 로그에 'press_duration'이 없다면 수집된 데이터의 timestamp 차이를 활용
                
                # (약식 구현) timestamp의 diff 중 사람이 누를법한 시간대만 추출
                diffs = df['timestamp'].diff().dropna()
                valid_diffs = diffs[(diffs >= 0.02) & (diffs <= 0.15)] 
                durations.extend(valid_diffs.tolist())
                        
            except Exception: pass

        if len(durations) > 100:
            # 이상치 제거 (IQR)
            q1, q3 = np.percentile(durations, [25, 75])
            iqr = q3 - q1
            filtered = [x for x in durations if q1 - 1.5*iqr <= x <= q3 + 1.5*iqr]
            
            if filtered:
                self.p_mean = np.mean(filtered)
                self.p_std = np.std(filtered)
                self.is_fitted = True
                print(f"✅ 내 습관 학습 완료: 평균 {self.p_mean*1000:.1f}ms (±{self.p_std*1000:.1f}ms)")
                print(f"🎭 익명화(Blending) 비율: {self.blending_ratio*100}% 본인 + {(1-self.blending_ratio)*100}% 일반인")
        else:
            print("⚠️ 데이터 부족: 일반인 모델 위주로 동작합니다.")

    def _sample_ex_gaussian(self, mu, sigma, tau):
        """Ex-Gaussian 분포에서 샘플링 (Normal + Exponential)"""
        # 정규분포 성분 (기본 반응)
        normal_component = random.gauss(mu, sigma)
        # 지수분포 성분 (인지 처리 지연, Long Tail)
        exponential_component = random.expovariate(1.0 / tau)
        return normal_component + exponential_component

    def get_press_duration(self):
        """
        혼합 모델(Mixture Model)을 통해 키 누름 시간 반환
        """
        # 확률적으로 소스 선택 (Mixture)
        if self.is_fitted and random.random() < self.blending_ratio:
            # [A] 내 습관대로 누름 (Fingerprint)
            duration = random.gauss(self.p_mean, self.p_std)
        else:
            # [B] 일반적인 사람처럼 누름 (Ex-Gaussian, Fingerprint Masking)
            duration = self._sample_ex_gaussian(self.g_mu, self.g_sigma, self.g_tau)
        
        # 물리적 한계 (최소 20ms ~ 최대 150ms)
        return max(0.02, min(0.15, duration))