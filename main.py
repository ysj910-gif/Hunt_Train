# main.py
import tkinter as tk
from gui import MapleHunterUI
import ctypes

# === 안전한 DPI 설정 (충돌 방지) ===
try:
    # 윈도우 10 이상에서 가장 호환성이 좋은 설정
    ctypes.windll.shcore.SetProcessDpiAwareness(1) 
except AttributeError:
    try:
        # 구버전 윈도우 호환
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass
# =================================

if __name__ == "__main__":
    root = tk.Tk()
    
    # 앱 실행 시 창을 잠시 맨 앞으로 가져옴
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)
    
    app = MapleHunterUI(root)
    root.mainloop()