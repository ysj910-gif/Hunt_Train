# platform_manager.py 파일의 내용으로 아래 코드를 넣으세요
import json

class PlatformManager:
    def __init__(self):
        self.platforms = []

    def load_platforms(self, file_path):
        """맵 에디터에서 만든 JSON 파일을 불러옵니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.platforms = data.get('platforms', [])
                print(f"성공적으로 {len(self.platforms)}개의 발판을 불러왔습니다.")
        except Exception as e:
            print(f"발판 로딩 실패: {e}")

    def get_current_platform(self, char_x, char_y):
        """캐릭터 좌표를 기반으로 현재 밟고 있는 발판을 찾습니다."""
        for p in self.platforms:
            if p['x_start'] <= char_x <= p['x_end'] and abs(char_y - p['y']) < 10:
                return p
        return None
    
    def get_distances(self, char_x, char_y):
        distances = {'up': None, 'down': None, 'left': None, 'right': None}
    
        for p in self.platforms:
        # 1. 상하 거리 (캐릭터의 X 좌표가 발판 범위 내에 있을 때)
            if p['x_start'] <= char_x <= p['x_end']:
                diff_y = p['y'] - char_y
                if diff_y > 0: # 아래쪽 발판
                    if distances['down'] is None or diff_y < distances['down']:
                        distances['down'] = diff_y
                elif diff_y < 0: # 위쪽 발판
                    if distances['up'] is None or abs(diff_y) < distances['up']:
                        distances['up'] = abs(diff_y)
        
        # 2. 좌우 거리 (캐릭터와 발판의 Y 좌표가 비슷할 때, 오차 15px 허용)
            if abs(p['y'] - char_y) < 15:
                if p['x_end'] < char_x: # 왼쪽 발판
                    diff_x = char_x - p['x_end']
                    if distances['left'] is None or diff_x < distances['left']:
                        distances['left'] = diff_x
                elif p['x_start'] > char_x: # 오른쪽 발판
                    diff_x = p['x_start'] - char_x
                    if distances['right'] is None or diff_x < distances['right']:
                        distances['right'] = diff_x

        return distances