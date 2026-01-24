# modules/job_manager.py

import json
import os

class JobManager:
    def __init__(self, filepath="jobs.json"):
        self.filepath = filepath
        self.job_map = self._load_jobs()

    def _load_jobs(self):
        """파일에서 직업 목록을 불러옴 (없으면 빈 딕셔너리 생성)"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def get_job_id(self, job_name):
        """
        직업 이름을 주면 ID를 반환.
        만약 처음 보는 직업이라면? -> 자동으로 새 ID 부여하고 저장!
        """
        if job_name not in self.job_map:
            # 새로운 직업 발견!
            new_id = len(self.job_map) # 0부터 순차 증가 (0, 1, 2...)
            self.job_map[job_name] = new_id
            self._save_jobs()
            print(f"🆕 새로운 직업 등록: {job_name} (ID: {new_id})")
            
        return self.job_map[job_name]

    def _save_jobs(self):
        """직업 목록을 파일에 저장 (영구 보존)"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.job_map, f, indent=4)
            
    def get_all_jobs(self):
        """GUI 콤보박스용: 등록된 모든 직업 이름 리스트 반환"""
        return list(self.job_map.keys())

    def get_num_jobs(self):
        """모델 설정용: 현재 등록된 직업 수 반환"""
        return len(self.job_map)