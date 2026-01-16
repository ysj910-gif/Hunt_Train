# utils.py
import json
import os
import config

def load_config():
    """설정 파일 로드 (없으면 기본값 반환)"""
    if os.path.exists(config.CONFIG_FILE):
        try:
            with open(config.CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(data):
    """설정 파일 저장"""
    try:
        with open(config.CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"저장 실패: {e}")
        return False