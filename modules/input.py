import time
import random
from pynput.keyboard import Controller, Key, KeyCode

class InputHandler:
    def __init__(self):
        self.keyboard = Controller()
        self.key_map = {}
        self.held_keys = set()
        
        # [수정] 0.05 -> 0.1로 변경 (초당 10회 입력 제한)
        # 너무 빠르면 키가 씹히거나 게임이 인식을 못함
        self.MIN_KEY_INTERVAL = 0.10
        
        self.PRESS_DURATION_MIN = 0.05
        self.PRESS_DURATION_MAX = 0.08
        
        self.last_press_time = {}

    def update_key_map(self, new_map):
        self.key_map = new_map

    def get_pynput_key(self, key_name):
        key_name = key_name.lower()
        special_keys = {
            'space': Key.space, 'enter': Key.enter, 'esc': Key.esc,
            'shift': Key.shift, 'ctrl': Key.ctrl, 'alt': Key.alt,
            'tab': Key.tab, 'up': Key.up, 'down': Key.down,
            'left': Key.left, 'right': Key.right,
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'ins': Key.insert, 'del': Key.delete, 'home': Key.home, 'end': Key.end,
            'pgup': Key.page_up, 'pgdn': Key.page_down
        }
        if key_name in special_keys: return special_keys[key_name]
        elif len(key_name) == 1: return KeyCode.from_char(key_name)
        return None

    def press(self, key_name):
        """단발성 입력 (Tap) - 쿨타임 적용됨"""
        target_key = self.get_pynput_key(key_name)
        if not target_key: return

        now = time.time()
        last_time = self.last_press_time.get(key_name, 0)
        
        # 쿨타임 체크 (너무 빠른 연타 방지)
        if now - last_time < self.MIN_KEY_INTERVAL:
            return 

        self.keyboard.press(target_key)
        self.last_press_time[key_name] = now
        
        # 꾹 눌렀다 떼는 시간 (사람처럼)
        time.sleep(random.uniform(self.PRESS_DURATION_MIN, self.PRESS_DURATION_MAX)) 
        self.keyboard.release(target_key)

    def hold(self, key_name):
        """지속 입력 (Hold)"""
        if key_name in self.held_keys: return
        target_key = self.get_pynput_key(key_name)
        if target_key:
            self.keyboard.press(target_key)
            self.held_keys.add(key_name)
            self.last_press_time[key_name] = time.time()

    def release(self, key_name):
        """키 떼기"""
        if key_name not in self.held_keys: return
        target_key = self.get_pynput_key(key_name)
        if target_key:
            self.keyboard.release(target_key)
            self.held_keys.discard(key_name)
            time.sleep(0.01)

    def release_all(self):
        for k in list(self.held_keys): self.release(k)
        self.held_keys.clear()