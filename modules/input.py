# modules/input.py
import serial
import time
import config

class InputHandler:
    def __init__(self):
        self.key_map = {}
        self.ser = None
        self.connect_arduino()

    def connect_arduino(self):
        """아두이노와 시리얼 연결 시도"""
        try:
            self.ser = serial.Serial(
                port=config.SERIAL_PORT,
                baudrate=config.BAUD_RATE,
                timeout=0.1
            )
            time.sleep(2) # 아두이노 리셋 대기
            print(f"✅ 아두이노 연결 성공: {config.SERIAL_PORT}")
        except Exception as e:
            print(f"❌ 아두이노 연결 실패: {e}")
            self.ser = None

    def update_key_map(self, new_map):
        self.key_map = new_map

    def send_cmd(self, command):
        """아두이노로 원시 명령어 전송 (예: 'Pspace')"""
        if self.ser and self.ser.is_open:
            # 명령어 끝에 개행문자(\n)를 붙여서 전송
            self.ser.write(f"{command}\n".encode())
        else:
            print(f"[Simulation] Serial Send: {command}")

    def execute(self, action_name):
        """추상 행동 -> 실제 키 입력 (Press & Release)"""
        real_key = self.key_map.get(action_name)
        
        if not real_key:
            return f"No Key for {action_name}"

        # 1. 키 누르기 (Press)
        self.send_cmd(f"P{real_key}")
        
        # 2. 아주 짧은 대기 (키 씹힘 방지, 30ms~50ms)
        time.sleep(0.05)
        
        # 3. 키 떼기 (Release)
        self.send_cmd(f"R{real_key}")
        
        return f"Hardware Input: [{real_key}]"

    def hold(self, action_name):
        """키 꾹 누르고 있기 (이동용)"""
        real_key = self.key_map.get(action_name)
        if real_key:
            self.send_cmd(f"P{real_key}")

    def release(self, action_name):
        """키 떼기"""
        real_key = self.key_map.get(action_name)
        if real_key:
            self.send_cmd(f"R{real_key}")
            
    def release_all(self):
        """비상 정지"""
        self.send_cmd("S")