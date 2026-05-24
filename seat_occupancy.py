"""SeatOccupancy — 좌석별 점유 점수 + 카페 컨텍스트 산출.

페이즈 6 재설계: 점수 축을 두 개로 분리한다.
  1) 점유 흔적 (OCCUPANCY_WEIGHTS) — 가방·노트북·책. 주인이 자리에 있다는 증거.
     이게 status (OCCUPIED/PARTIAL/ABANDONED)를 결정.
  2) 카페 컨텍스트 (CAFE_CONTEXT_CLASSES) — 컵. 분실 대상이 아니라 도메인 신호.
     별도로 _last_cup_seen에 기록 → is_cafe_active()로 조회.

페이즈 6-A: 추적 대상이 가방·노트북·책으로 확장 (TRACKABLE_CLASSES).
  이중 카운트 방지 — 추적 중인 물건은 tracked_items 루프로만 가산하고,
  seat_indicators 루프에서는 TRACKABLE_CLASSES를 건너뜀.
  추적 자체가 짧은 누락을 흡수하므로 별도 indicator persistence는 더 이상
  필요 없음 (페이즈 5에서 도입했던 _last_seen_indicator 제거).

객체 인식 끊김 보강: SMOOTH_WINDOW=30 (~1초) 점수 이동평균.

의존성: config(상수), seats(Seat / box_in_seat). leaf 모듈만 참조.
"""
from collections import deque

from config import (
    OCCUPANCY_WEIGHTS,
    TRACKABLE_CLASSES,
    CAFE_CONTEXT_CLASSES,
    CUP_PRESENCE_WINDOW,
    SEAT_WEIGHT_PERSON,
    SEAT_OCCUPIED_THRESHOLD,
    SEAT_PARTIAL_THRESHOLD,
    SEAT_INDICATOR_NAMES,
)
from seats import Seat, box_in_seat


# 클래스가 OCCUPANCY_WEIGHTS에 없을 때 폴백 (예: cls=None인 트랙).
_DEFAULT_ITEM_WEIGHT = 30


def _display_name(cls: int | None) -> str:
    """breakdown 표시용 이름. 가방류(24/26/28)는 'bag'으로 묶음."""
    if cls in (24, 26, 28):
        return "bag"
    if cls is None:
        return "item"
    return SEAT_INDICATOR_NAMES.get(cls, str(cls))


class SeatOccupancy:
    """좌석별 점유 점수, 상태 분류, 카페 컨텍스트."""

    SMOOTH_WINDOW = 30   # ~1초 (FRAME_SKIP=2, 30fps 소스 가정)

    def __init__(self, seats: list[Seat]):
        self._seats = seats
        # 현재 프레임의 raw 점수 + breakdown (디버그용).
        self._raw_scores: dict[str, float] = {s.seat_id: 0.0 for s in seats}
        self._breakdowns: dict[str, dict[str, float]] = {
            s.seat_id: {} for s in seats
        }
        # 이동평균용 raw score 이력.
        self._score_history: dict[str, deque] = {
            s.seat_id: deque(maxlen=self.SMOOTH_WINDOW) for s in seats
        }
        # 카페 컨텍스트 — seat_id → 컵 마지막으로 본 시각.
        self._last_cup_seen: dict[str, float] = {}

    # ── 매 프레임 호출 ────────────────────────────────────
    def update(self, person_boxes, tracked_items, seat_indicators, current_time):
        """좌석별 점수·카페 컨텍스트 갱신.

        Parameters
        ----------
        person_boxes      : list[(x1, y1, x2, y2)]
        tracked_items     : dict[master_id, TrackedItem]  — 활성 트랙만
        seat_indicators   : list[(x1, y1, x2, y2, cls)]  — 추적되지 않는 raw 표식
                            (현재는 컵만 실제 흐름. TRACKABLE_CLASSES는 여기 안 옴)
        current_time      : float — 카페 컨텍스트 타임스탬프
        """
        for seat in self._seats:
            sid = seat.seat_id
            if seat.region == (0, 0, 0, 0):
                self._raw_scores[sid] = 0.0
                self._breakdowns[sid] = {}
                self._score_history[sid].append(0.0)
                continue

            # ── 1) 카페 컨텍스트: 컵이 좌석 안에 있는가 ──────
            cup_in_seat = any(
                cls in CAFE_CONTEXT_CLASSES and box_in_seat((x1, y1, x2, y2), seat)
                for x1, y1, x2, y2, cls in seat_indicators
            )
            if cup_in_seat:
                self._last_cup_seen[sid] = current_time

            # ── 2) 점유 흔적 breakdown ──────────────────────
            breakdown: dict[str, float] = {}

            # 추적되지 않는 OCCUPANCY 표식 (방어용 — 현재 흐름엔 없음).
            for x1, y1, x2, y2, cls in seat_indicators:
                if cls in CAFE_CONTEXT_CLASSES:
                    continue   # 컵은 별도 축
                if cls in TRACKABLE_CLASSES:
                    continue   # 추적 중이라 별도 회계
                weight = OCCUPANCY_WEIGHTS.get(cls, 0)
                if weight == 0:
                    continue
                if box_in_seat((x1, y1, x2, y2), seat):
                    name = _display_name(cls)
                    breakdown[name] = breakdown.get(name, 0) + weight

            # 추적 중인 물건 — TRACKABLE_CLASSES 가중치 직접 가산.
            for item in tracked_items.values():
                if item.seat_id != sid:
                    continue
                cls = getattr(item, "cls", None)
                weight = OCCUPANCY_WEIGHTS.get(cls, _DEFAULT_ITEM_WEIGHT)
                name = _display_name(cls)
                breakdown[name] = breakdown.get(name, 0) + weight

            # 사람 — 좌석 안에 한 명이라도 있으면 가중치 1회만.
            person_present = any(box_in_seat(pb, seat) for pb in person_boxes)
            breakdown["person"] = SEAT_WEIGHT_PERSON if person_present else 0

            score = sum(breakdown.values())
            self._raw_scores[sid] = score
            self._breakdowns[sid] = breakdown
            self._score_history[sid].append(score)

    # ── 점유 흔적 점수 ────────────────────────────────────
    def get_score(self, seat_id: str) -> float:
        """이동평균(smoothed) 점수. 의사결정·VLM에 사용."""
        hist = self._score_history.get(seat_id)
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def get_score_raw(self, seat_id: str) -> float:
        """현재 프레임의 즉시 점수 (디버그용)."""
        return self._raw_scores.get(seat_id, 0.0)

    def get_score_raw_range(self, seat_id: str) -> tuple[float, float]:
        """스무딩 window의 (가장 오래된, 가장 최신) raw 점수."""
        hist = self._score_history.get(seat_id)
        if not hist:
            return (0.0, 0.0)
        return (hist[0], hist[-1])

    def get_status(self, seat_id: str) -> str:
        """점유 흔적 점수 기반 상태. 카페 컨텍스트는 별도 메서드."""
        score = self.get_score(seat_id)
        if score >= SEAT_OCCUPIED_THRESHOLD:
            return "OCCUPIED"
        if score >= SEAT_PARTIAL_THRESHOLD:
            return "PARTIAL"
        return "ABANDONED"

    def get_breakdown(self, seat_id: str) -> dict[str, float]:
        """현재 프레임의 가중치 분해 (raw, 디버그용)."""
        return dict(self._breakdowns.get(seat_id, {}))

    # ── 카페 컨텍스트 ─────────────────────────────────────
    def is_cafe_active(self, seat_id: str | None, current_time: float) -> bool:
        """컵이 최근 CUP_PRESENCE_WINDOW 초 이내에 보였으면 True."""
        if seat_id is None:
            return False
        last = self._last_cup_seen.get(seat_id)
        if last is None:
            return False
        return (current_time - last) < CUP_PRESENCE_WINDOW

    def cup_disappeared_recently(self, seat_id: str | None, current_time: float) -> bool:
        """컵이 직전엔 있었는데 최근 30~60초 사이에 사라졌는지.
        손님 퇴장 신호 감지용 (페이즈 7에서 정교화 예정 — 일단 단순 구현)."""
        if seat_id is None:
            return False
        last = self._last_cup_seen.get(seat_id)
        if last is None:
            return False
        elapsed = current_time - last
        return CUP_PRESENCE_WINDOW <= elapsed < 60

    # ── 물건 좌석 매핑 ────────────────────────────────────
    def item_to_seat(self, item_bbox) -> str | None:
        """물건 박스 중심점이 속한 좌석 ID. 없으면 None."""
        for seat in self._seats:
            if seat.region == (0, 0, 0, 0):
                continue
            if box_in_seat(item_bbox, seat):
                return seat.seat_id
        return None
