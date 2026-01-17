import tkinter as tk
from collector_gui import DataCollectorApp

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.mainloop()