r"""좌석 영역 좌표 측정 스크립트.

사용법:
    python define_seats.py C:\Users\tjwns\my_vlm_project\case1.mp4

조작:
    - 마우스 드래그: 좌석 영역 사각형 그리기
    - s: 그린 영역 저장 (좌석 ID 입력)
    - n: 다음 프레임으로 (10초 점프)
    - p: 이전 프레임으로 (10초 점프)
    - r: 현재 그린 영역 초기화
    - u: 마지막 저장 좌석 취소
    - q: 종료 + seats.py에 붙여넣을 코드 출력
"""
import sys
import cv2

WINDOW = "Define Seats — drag to draw, s=save, n/p=jump, q=quit"

drawing = False
ix, iy = -1, -1
cur_box = None  # 현재 그리는 중인 박스
seats = []  # [(seat_id, (x1, y1, x2, y2)), ...]


def on_mouse(event, x, y, flags, _):
    global drawing, ix, iy, cur_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cur_box = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cur_box = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        cur_box = (x1, y1, x2, y2) if (x2 - x1) > 10 and (y2 - y1) > 10 else None


def draw_overlay(frame):
    out = frame.copy()
    # 이미 저장한 좌석들 (초록)
    for sid, (x1, y1, x2, y2) in seats:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, sid, (x1 + 6, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    # 현재 그리는 중인 박스 (노랑)
    if cur_box:
        x1, y1, x2, y2 = cur_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 220), 2)
    # 상태 표시
    h = out.shape[0]
    cv2.putText(out, f"Seats: {len(seats)} | drag=draw, s=save, n/p=jump, u=undo, q=quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def main():
    if len(sys.argv) < 2:
        print("사용법: python define_seats.py path/to/video.mp4")
        sys.exit(1)

    global cur_box
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        print(f"영상을 열 수 없음: {sys.argv[1]}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    jump = int(fps * 10)  # 10초 점프

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)
    cv2.setMouseCallback(WINDOW, on_mouse)

    cur_pos = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
        ok, frame = cap.read()
        if not ok:
            cur_pos = max(0, cur_pos - jump)
            continue

        while True:
            cv2.imshow(WINDOW, draw_overlay(frame))
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print_result()
                return

            elif key == ord('s'):
                if not cur_box:
                    print("⚠ 먼저 마우스로 영역을 그리세요.")
                    continue
                sid = input("좌석 ID (예: A, B, C): ").strip()
                if sid:
                    seats.append((sid, cur_box))
                    print(f"✔ 좌석 {sid} 저장됨: {cur_box}")
                    cur_box = None

            elif key == ord('u'):
                if seats:
                    removed = seats.pop()
                    print(f"↩ 좌석 {removed[0]} 취소")

            elif key == ord('r'):
                cur_box = None

            elif key == ord('n'):
                cur_pos = min(total - 1, cur_pos + jump)
                break

            elif key == ord('p'):
                cur_pos = max(0, cur_pos - jump)
                break


def print_result():
    if not seats:
        print("\n저장된 좌석 없음.")
        return
    print("\n" + "=" * 60)
    print("seats.py의 SEATS 리스트를 아래로 교체:")
    print("=" * 60)
    print("SEATS: list[Seat] = [")
    for sid, (x1, y1, x2, y2) in seats:
        print(f'    Seat("{sid}", ({x1}, {y1}, {x2}, {y2})),')
    print("]")
    print("=" * 60)


if __name__ == "__main__":
    main()