import json
import tkinter as tk
from tkinter import filedialog, simpledialog
import os

def shift_map_y():
    # 윈도우 창 숨기기
    root = tk.Tk()
    root.withdraw()

    print("Step 1. 좌표를 수정할 맵 파일(.json)을 선택하세요.")
    file_path = filedialog.askopenfilename(title="맵 JSON 파일 선택", filetypes=[("JSON", "*.json")])
    if not file_path:
        print("❌ 파일이 선택되지 않았습니다.")
        return

    # 사용자로부터 이동할 Y값 입력 받기 (기본값 -44)
    # 제목 표시줄 높이만큼 맵 좌표를 '위로' 올려야 하므로 음수 입력
    shift_val = simpledialog.askinteger("Y좌표 이동", "Y좌표를 얼마나 이동할까요?\n(제목표시줄 제거 시 보통 -44)", initialvalue=-44)
    
    if shift_val is None:
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n🔄 모든 Y좌표를 {shift_val}만큼 이동합니다...")

        # 1. 발판 (Platforms)
        count_plat = 0
        for p in data.get('platforms', []):
            p['y'] += shift_val
            count_plat += 1

        # 2. 사다리 (Ropes)
        count_rope = 0
        for r in data.get('ropes', []):
            r['y_top'] += shift_val
            r['y_bottom'] += shift_val
            count_rope += 1

        # 3. 몬스터/스폰 (Mobs)
        count_mob = 0
        for m in data.get('mobs', []):
            if 'y' in m:
                m['y'] += shift_val
                count_mob += 1

        # 파일 저장
        folder, filename = os.path.split(file_path)
        new_filename = f"Shifted_{filename}"
        save_path = os.path.join(folder, new_filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        print(f"✅ 수정 완료!")
        print(f"   - 발판 {count_plat}개, 사다리 {count_rope}개, 객체 {count_mob}개 수정됨")
        print(f"   - 저장된 파일: {save_path}")
        print("\n👉 [중요] 이제 gui.py에서 이 파일을 로드하고, Y Offset을 0으로 설정하세요.")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    shift_map_y()