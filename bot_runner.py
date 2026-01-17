# bot_runner.py
import time
import threading
import pandas as pd
import joblib
import config
from modules.input import InputHandler

class BotRunner:
    def __init__(self):
        self.model = None
        self.is_running = False
        self.thread = None
        self.vision = None
        
        self.input_handler = InputHandler()
        self.pressed_keys = set()
        
        self.offset_x = 0
        self.offset_y = 0

    def load_model(self, path):
        try:
            self.model = joblib.load(path)
            return True, "모델 로드 성공"
        except Exception as e:
            return False, f"모델 로드 실패: {e}"

    def start(self, vision_engine, offset_x=0, offset_y=0):
        if self.is_running: return
        if self.model is None:
            raise Exception("모델이 준비되지 않았습니다.")
            
        self.vision = vision_engine
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.is_running = True
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"▶️ 봇 시작 (Offset: {offset_x}, {offset_y})")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.release_all_keys()
        print("⏹️ 봇 중지")

    def update_key_state(self, action_str):
        if action_str == 'None' or pd.isna(action_str):
            target_keys = set()
        else:
            target_keys = set(action_str.split('+'))

        # Release
        for k in list(self.pressed_keys):
            if k not in target_keys:
                self.input_handler.release(k)
                self.pressed_keys.remove(k)
        
        # Press
        for k in target_keys:
            if k not in self.pressed_keys:
                self.input_handler.hold(k)
                self.pressed_keys.add(k)

    def release_all_keys(self):
        self.input_handler.release_all()
        self.pressed_keys.clear()

    def _loop(self):
        while self.is_running:
            try:
                # VisionSystem에서 화면과 정보를 한 번에 받아옴
                frame, entropy, kill_count, raw_px, raw_py = self.vision.capture_and_analyze()

                # 화면이 안 잡혔거나(검은색 0,0,0) 캐릭터를 못 찾은 경우(0,0) 대기
                if frame is None or frame.size == 0:
                    time.sleep(0.5)
                    continue

                # 봇 로직 실행
                player_x = raw_px - self.offset_x
                player_y = raw_py - self.offset_y
                current_platform_id = -1 

                features = pd.DataFrame(
                    [[player_x, player_y, entropy, current_platform_id]], 
                    columns=['player_x', 'player_y', 'entropy', 'platform_id']
                )

                action = self.model.predict(features)[0]
                self.update_key_state(action)

                time.sleep(0.05) 

            except Exception as e:
                print(f"❌ Bot Loop Error: {e}")
                time.sleep(1)
        
        self.release_all_keys()