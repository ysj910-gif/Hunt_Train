# config.py

# 파일 경로
CONFIG_FILE = "hunter_config.json"

# 기본 캡처 영역 (x, y, w, h)
DEFAULT_CAPTURE_AREA = {"top": 100, "left": 100, "width": 640, "height": 360}

# 기본 스킬 쿨타임 (초)
DEFAULT_COOLDOWNS = {
    "ultimate": 30.0,
    "sub_attack": 8.0,
    "buff": 180.0,
    "rope": 0.0,
    "jump": 0.0,
    "attack": 0.0
}

# 기본 키 매핑
DEFAULT_KEY_MAP = {
    "ultimate": "r",
    "sub_attack": "a",
    "buff": "1",
    "rope": "c",
    "jump": "space",
    "attack": "ctrl"
}

# config.py

# ... (기존 설정들) ...

# [하드웨어 통신 설정]
# 윈도우: "COM3", "COM4" 등 (장치 관리자에서 확인)
# 맥/리눅스: "/dev/ttyUSB0" 등
SERIAL_PORT = "COM9"  
BAUD_RATE = 115200