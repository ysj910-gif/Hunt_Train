# main.py
import tkinter as tk
from gui import MapleHunterUI

# === [추가할 코드 시작] 듀얼 모니터/DPI 문제 해결 ===
import ctypes
try:
    # 윈도우 8.1 이상
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    # 윈도우 8 이하
    ctypes.windll.user32.SetProcessDPIAware()
# === [추가할 코드 끝] ==============================

if __name__ == "__main__":
    root = tk.Tk()
    app = MapleHunterUI(root)
    root.mainloop()