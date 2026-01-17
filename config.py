import numpy as np

# 기본 데이터 저장 경로 및 설정 (GUI에서 변경 가능)
DATA_DIR = "training_data_v2"
SAVE_NAME = "recording.csv"
JOB = "전사"
MOVE_TYPE = "DJ" # DJ: 더블점프, TP: 텔레포트
TARGET_RECORD_TIME = 600
FPS_LIMIT = 10

# 미니맵 ROI (창 내부에서의 상대 좌표: left, top, width, height)
# 초기값은 0이며 GUI의 '미니맵 영역 지정' 버튼으로 설정함
MINIMAP_ROI = {'left': 0, 'top': 0, 'width': 0, 'height': 0}

# 캐릭터 탐지 색상 (노란색 점)
YELLOW_HSV_LOWER = np.array([25, 150, 150])
YELLOW_HSV_UPPER = np.array([35, 255, 255])

# 조작키 매핑
KEY_ACTION_MAP = {}