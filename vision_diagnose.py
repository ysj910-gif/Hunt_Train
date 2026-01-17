import cv2
import mss
import numpy as np
import win32gui
import win32ui
import win32con
import ctypes
from ctypes import wintypes
import os

# === 설정 ===
WINDOW_TITLE = "MapleStory"
SAVE_DIR = "diagnose_result"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# === 1. 관리자 권한 확인 ===
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# === 2. 윈도우 좌표 가져오기 (Client Area vs Window Area) ===
def get_window_info(hwnd):
    # 전체 창 크기 (타이틀바 포함)
    rect = win32gui.GetWindowRect(hwnd)
    x, y, r, b = rect
    w = r - x
    h = b - y
    
    # 실제 게임 화면 크기 (Client Area)
    client_rect = win32gui.GetClientRect(hwnd)
    cw = client_rect[2]
    ch = client_rect[3]
    
    # Client Area의 스크린 좌표 계산
    pt = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    cx, cy = pt.x, pt.y
    
    return {
        "hwnd": hwnd,
        "full": (x, y, w, h),
        "client": (cx, cy, cw, ch)
    }

# === 3. 캡처 방식 테스트 ===

def test_mss(info):
    print("📸 [Test 1] MSS 캡처 시도...")
    try:
        cx, cy, cw, ch = info['client']
        # mss는 딕셔너리 형태로 영역 지정
        monitor = {"top": cy, "left": cx, "width": cw, "height": ch}
        
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(f"{SAVE_DIR}/1_mss_capture.png", frame)
            print(f"   ✅ 저장 완료: {SAVE_DIR}/1_mss_capture.png")
            return True
    except Exception as e:
        print(f"   ❌ MSS 실패: {e}")
        return False

def test_win32_bitblt(info):
    print("📸 [Test 2] Win32 BitBlt 캡처 시도...")
    try:
        hwnd = info['hwnd']
        w, h = info['client'][2], info['client'][3]
        
        # DC 준비
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        
        # 캡처 (타이틀바 제외하고 Client 영역만 찍기 위해 오프셋 조정이 필요할 수 있으나, 일단 WindowDC 기준)
        # 정확히는 ClientToScreen 좌표 차이를 이용해야 하지만, 테스트용이므로 (0,0) 시도
        # 보통 BitBlt는 화면이 가려지면 검게 나옵니다.
        saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
        
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (h, w, 4)
        
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(f"{SAVE_DIR}/2_win32_bitblt.png", frame)
        print(f"   ✅ 저장 완료: {SAVE_DIR}/2_win32_bitblt.png")
        return True
    except Exception as e:
        print(f"   ❌ Win32 BitBlt 실패: {e}")
        return False

def test_printwindow(info):
    print("📸 [Test 3] PrintWindow (RenderFullContent) 캡처 시도...")
    try:
        hwnd = info['hwnd']
        w, h = info['full'][2], info['full'][3] # PrintWindow는 전체 창 기준
        
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        
        # PW_RENDERFULLCONTENT = 2 (윈도우 8.1 이상)
        ret = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
        
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (h, w, 4)
        
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(f"{SAVE_DIR}/3_printwindow.png", frame)
        print(f"   ✅ 저장 완료: {SAVE_DIR}/3_printwindow.png")
        return True
    except Exception as e:
        print(f"   ❌ PrintWindow 실패: {e}")
        return False

# === 메인 실행 ===
if __name__ == "__main__":
    print("=== 메이플스토리 화면 캡처 진단 도구 ===")
    
    if not is_admin():
        print("⚠️ 경고: 관리자 권한이 없습니다. 캡처가 차단될 수 있습니다.")
        print("   -> CMD를 관리자 권한으로 실행 후 다시 시도하세요.")
    else:
        print("✅ 관리자 권한 확인됨.")
        
    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
    if not hwnd:
        print(f"❌ '{WINDOW_TITLE}' 창을 찾을 수 없습니다. 게임을 실행해 주세요.")
    else:
        print(f"✅ 창 발견! HWND: {hwnd}")
        
        # 창 정보 출력
        info = get_window_info(hwnd)
        print(f"   - 전체 좌표 (Left, Top, W, H): {info['full']}")
        print(f"   - 실제 화면 (Left, Top, W, H): {info['client']}")
        
        if info['client'][2] == 0 or info['client'][3] == 0:
            print("⚠️ 경고: 클라이언트 영역 크기가 0입니다. 창이 최소화되어 있거나 로딩 중일 수 있습니다.")
        
        # 테스트 수행
        test_mss(info)
        test_win32_bitblt(info)
        test_printwindow(info)
        
        print("\n=== 진단 완료 ===")
        print(f"📂 '{SAVE_DIR}' 폴더에 저장된 3개의 이미지를 확인하세요.")
        print("1. 1_mss_capture.png -> 이것이 잘 나오면 mss 방식 사용 (bot_runner 수정 필요)")
        print("2. 2_win32_bitblt.png -> 이것이 잘 나오면 win32gui 사용")
        print("3. 3_printwindow.png -> 이것만 잘 나오면 PrintWindow 사용")
        print("4. 모두 검은색이면 -> 게임 보안 프로그램(NGS)이 차단 중이거나 전체화면 모드 문제")