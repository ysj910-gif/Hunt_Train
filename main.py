# main.py
import tkinter as tk
from gui import MapleHunterUI
import ctypes

if __name__ == "__main__":
    root = tk.Tk()
    
    # 앱 실행 시 창을 잠시 맨 앞으로 가져옴
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)
    
    app = MapleHunterUI(root)
    root.mainloop()