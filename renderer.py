"""OverlayRenderer — 프레임 위 시각화 (바운딩 박스, 라벨, HUD).

TrackedItem은 type hint 용도로만 참조하므로 TYPE_CHECKING으로 import 분리
(런타임에는 duck-typed 속성만 사용 — item.state, item.score, item.elapsed 등).
"""
from typing import TYPE_CHECKING
import time
import cv2

from config import (
    STATE_COLORS, SAFE_DISTANCE, ST_TRACKING,
    SEAT_COLOR_OCCUPIED, SEAT_COLOR_PARTIAL, SEAT_COLOR_ABANDONED,
)

if TYPE_CHECKING:
    from tracked_item import TrackedItem


class OverlayRenderer:
    """프레임 위에 추적 정보를 그리는 전담 클래스"""

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

    @staticmethod
    def draw_item(frame, x1, y1, x2, y2, item: "TrackedItem"):
        """추적 물건 바운딩 박스 + 상태 라벨 + 경과 시간"""
        color = STATE_COLORS.get(item.state, (200, 200, 200))
        thickness = 2 if item.state == ST_TRACKING else 3

        # 바운딩 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 상태 라벨 배경
        elapsed_sec = int(item.elapsed)
        label = f"ID:{item.master_id} {item.state} {elapsed_sec}s"
        (tw, th), _ = cv2.getTextSize(label, OverlayRenderer.FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5),
                    OverlayRenderer.FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 점수 바 (박스 아래)
        bar_w = x2 - x1
        bar_h = 4
        filled = int(bar_w * item.score / 100)
        cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 2 + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 2 + bar_h), color, -1)

    @staticmethod
    def draw_person(frame, px1, py1, px2, py2, is_near_bag=False):
        """사람 바운딩 박스 (반투명)"""
        color = (255, 200, 0) if is_near_bag else (100, 100, 100)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)
        if is_near_bag:
            cv2.putText(frame, "NEAR", (px1, py1 - 5),
                        OverlayRenderer.FONT_SMALL, 1.0, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_seats(frame, seats, seat_occupancy):
        """좌석 영역을 점유 상태별 색으로 표시.

        OCCUPIED 초록 / PARTIAL 노랑 / ABANDONED 회색.
        영역 미설정(region=(0,0,0,0)) 좌석은 건너뜀.
        """
        if not seats:
            return
        color_map = {
            "OCCUPIED":  SEAT_COLOR_OCCUPIED,
            "PARTIAL":   SEAT_COLOR_PARTIAL,
            "ABANDONED": SEAT_COLOR_ABANDONED,
        }
        overlay = frame.copy()
        any_drawn = False
        for seat in seats:
            if seat.region == (0, 0, 0, 0):
                continue
            status = seat_occupancy.get_status(seat.seat_id)
            color  = color_map.get(status, (255, 0, 255))   # fallback 핑크
            sx1, sy1, sx2, sy2 = seat.region
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), color, -1)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
            label = f"seat_{seat.seat_id} [{status}]"
            cv2.putText(frame, label, (sx1 + 4, sy1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            any_drawn = True
        if any_drawn:
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    @staticmethod
    def draw_safe_radius(frame, cx, cy):
        """SAFE_DISTANCE 반경 표시 (점선 원)"""
        cv2.circle(frame, (cx, cy), SAFE_DISTANCE, (60, 60, 60), 1, cv2.LINE_AA)

    @staticmethod
    def draw_hud(frame, total_bags, total_persons):
        """화면 상단 HUD — 전체 통계"""
        h, w = frame.shape[:2]
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        info = f"  LOST ITEM DETECTION  |  Bags: {total_bags}  Persons: {total_persons}  |  {ts}"
        cv2.putText(frame, info, (8, 22),
                    OverlayRenderer.FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
