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
            print(f"❌ 아두이노 연결 실패: {e}\n(이 상태에서는 키 입력이 [Simulation]으로 출력만 됩니다.)")
            self.ser = None

    def update_key_map(self, new_map):
        self.key_map = new_map

    def send_cmd(self, command):
        """아두이노로 원시 명령어 전송 (예: 'Pspace')"""
        if self.ser and self.ser.is_open:
            try:
                # 명령어 끝에 개행문자(\n)를 붙여서 전송
                self.ser.write(f"{command}\n".encode())
            except Exception as e:
                print(f"⚠️ 전송 오류: {e}")
        else:
            # 연결 안 됐을 때 디버그용 출력
            pass 
            # print(f"[Simulation] Serial Send: {command}")

    def execute(self, action_name):
        """추상 행동 -> 실제 키 입력 (Press & Release)"""
        # 1. 매핑된 키가 있는지 확인
        real_key = self.key_map.get(action_name)
        
        # 2. 없으면 액션 이름 자체를 키로 사용 (예: "left", "space")
        if not real_key:
            real_key = action_name

        self.send_cmd(f"P{real_key}")
        time.sleep(0.05)
        self.send_cmd(f"R{real_key}")
        
        return f"Hardware Input: [{real_key}]"

    def hold(self, action_name):
        """키 꾹 누르고 있기"""
        # [수정] 매핑에 없으면 입력받은 키 그대로 사용
        real_key = self.key_map.get(action_name, action_name)
        if real_key:
            self.send_cmd(f"P{real_key}")

    def release(self, action_name):
        """키 떼기"""
        # [수정] 매핑에 없으면 입력받은 키 그대로 사용
        real_key = self.key_map.get(action_name, action_name)
        if real_key:
            self.send_cmd(f"R{real_key}")
            
    def release_all(self):
        """비상 정지"""
        self.send_cmd("S")