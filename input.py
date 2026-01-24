# modules/input.py
import serial
import time
import threading
import random
import atexit # [신규] 프로그램 종료 감지 모듈

class InputHandler:
    def __init__(self):
        # ==========================================
        # [설정] 아두이노 포트 (본인 환경에 맞게 수정)
        # ==========================================
        self.PORT = 'COM9'  
        self.BAUDRATE = 115200
        
        self.SERIAL_DELAY = 0.01  
        self.MIN_KEY_INTERVAL = 0.10
        self.PRESS_DURATION_MIN = 0.05
        self.PRESS_DURATION_MAX = 0.08
        
        self.ser = None
        self.key_map = {}
        self.held_keys = set()
        self.last_press_time = {}
        self.lock = threading.Lock()
        
        self.connect_arduino()

        # [핵심] 프로그램이 꺼질 때 자동으로 release_all 호출
        atexit.register(self.release_all)

    def connect_arduino(self):
        try:
            self.ser = serial.Serial(self.PORT, self.BAUDRATE, timeout=1)
            time.sleep(2) 
            print(f"✅ [Input] 아두이노 연결 성공 ({self.PORT})")
        except Exception as e:
            print(f"❌ [Input] 아두이노 연결 실패: {e}")

    def _send(self, command):
        if self.ser and self.ser.is_open:
            with self.lock:
                try:
                    msg = f"{command}\n"
                    self.ser.write(msg.encode())
                    self.ser.flush()
                    time.sleep(self.SERIAL_DELAY)
                except Exception as e:
                    print(f"⚠️ 전송 오류: {e}")

    def update_key_map(self, new_map):
        with self.lock:
            self.key_map = new_map

    def press(self, key_name):
        target = self.key_map.get(key_name, key_name).lower()
        now = time.time()
        last_time = self.last_press_time.get(key_name, 0)
        
        if now - last_time < self.MIN_KEY_INTERVAL: return

        self.last_press_time[key_name] = now
        self._send(f"P{target}")
        time.sleep(random.uniform(self.PRESS_DURATION_MIN, self.PRESS_DURATION_MAX))
        self._send(f"R{target}")
        time.sleep(0.02) 

    def hold(self, key_name):
        target = self.key_map.get(key_name, key_name).lower()
        if target not in self.held_keys:
            self._send(f"P{target}")
            self.held_keys.add(target)
            self.last_press_time[key_name] = time.time()

    def release(self, key_name):
        target = self.key_map.get(key_name, key_name).lower()
        if target in self.held_keys:
            self._send(f"R{target}")
            self.held_keys.remove(target)
        else:
            self._send(f"R{target}")

    def release_all(self):
        """모든 키 떼기 (최적화 버전)"""
        # [수정] 이미 키를 아무것도 안 누르고 있다면, 굳이 아두이노에 신호를 보내지 않음 (렉 방지)
        if not self.held_keys:
            return

        if self.ser and self.ser.is_open:
            try:
                # 확실하게 멈추기 위해 S 전송
                self.ser.write(b"S\n")
                self.ser.flush()
                time.sleep(0.01)
                
                self.held_keys.clear()
                # 로그가 너무 많이 뜨면 아래 줄을 주석(#) 처리하세요
                print("🛑 [Input] 모든 키 해제 완료")
            except Exception as e:
                print(f"⚠️ 정지 신호 전송 실패: {e}")
            
    # [신규] 특정 키만 빼고 다 떼기 (설치기 쓸 때 등)
    def release_all_except(self, keep_key):
        keep_key = self.key_map.get(keep_key, keep_key).lower()
        
        # 현재 누르고 있는 키들 중 keep_key가 아닌 건 모두 뗌
        # (복사본을 만들어서 순회해야 안전)
        for k in list(self.held_keys):
            if k != keep_key:
                self._send(f"R{k}")
                self.held_keys.remove(k)