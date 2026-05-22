"""SeatOccupancy — 좌석별 점유 점수 산출.

매 프레임 update()로 점수를 갱신하고, get_status / get_score / get_breakdown으로
조회한다. 의사결정(예: TRACKING → SUSPICIOUS 분기 조건)은 메인 파이프라인이
이 모듈의 결과를 읽어서 수행한다 — 이 모듈은 점수 계산만 책임진다.

의존성: config(상수), seats(Seat / box_in_seat). leaf 모듈만 참조.
"""
from config import (
    SEAT_WEIGHTS,
    SEAT_WEIGHT_PERSON,
    SEAT_OCCUPIED_THRESHOLD,
    SEAT_PARTIAL_THRESHOLD,
    SEAT_INDICATOR_NAMES,
)
from seats import Seat, box_in_seat


# 가방 클래스 가중치 — BAG_CLASSES(24/26/28)는 동일 가중치를 공유한다
# (TrackedBag는 class id를 보존하지 않으므로 단일 값으로 처리).
_BAG_WEIGHT = SEAT_WEIGHTS.get(24, 30)


class SeatOccupancy:
    """좌석별 점유 점수 계산 및 상태 분류."""

    def __init__(self, seats: list[Seat]):
        self._seats = seats
        self._scores: dict[str, float] = {s.seat_id: 0.0 for s in seats}
        self._breakdowns: dict[str, dict[str, float]] = {
            s.seat_id: {} for s in seats
        }

    # ── 매 프레임 호출 ────────────────────────────────────
    def update(self, person_boxes, tracked_bags, seat_indicators):
        """좌석별 점수를 새로 계산하여 내부 상태에 저장.

        Parameters
        ----------
        person_boxes      : list[(x1, y1, x2, y2)]
        tracked_bags      : dict[master_id, TrackedBag]  — 활성 트랙만
        seat_indicators   : list[(x1, y1, x2, y2, cls)]  — 매 프레임 raw 감지
        """
        for seat in self._seats:
            if seat.region == (0, 0, 0, 0):
                self._scores[seat.seat_id] = 0.0
                self._breakdowns[seat.seat_id] = {}
                continue

            breakdown: dict[str, float] = {}

            # 좌석 표식 (노트북·폰·책·컵)
            for x1, y1, x2, y2, cls in seat_indicators:
                if not box_in_seat((x1, y1, x2, y2), seat):
                    continue
                weight = SEAT_WEIGHTS.get(cls, 0)
                if weight == 0:
                    continue
                name = SEAT_INDICATOR_NAMES.get(cls, str(cls))
                breakdown[name] = breakdown.get(name, 0) + weight

            # 추적 중인 가방
            for bag in tracked_bags.values():
                if box_in_seat(bag._bbox, seat):
                    breakdown["bag"] = breakdown.get("bag", 0) + _BAG_WEIGHT

            # 사람 — 좌석 안에 한 명이라도 있으면 가중치 1회만 부여
            #        (여러 명 있어도 점유 신호 자체는 동일하다고 본다)
            person_present = any(
                box_in_seat(pb, seat) for pb in person_boxes
            )
            breakdown["person"] = SEAT_WEIGHT_PERSON if person_present else 0

            score = sum(breakdown.values())
            self._scores[seat.seat_id] = score
            self._breakdowns[seat.seat_id] = breakdown

    # ── 조회 ──────────────────────────────────────────────
    def get_score(self, seat_id: str) -> float:
        return self._scores.get(seat_id, 0.0)

    def get_status(self, seat_id: str) -> str:
        score = self._scores.get(seat_id, 0.0)
        if score >= SEAT_OCCUPIED_THRESHOLD:
            return "OCCUPIED"
        if score >= SEAT_PARTIAL_THRESHOLD:
            return "PARTIAL"
        return "ABANDONED"

    def get_breakdown(self, seat_id: str) -> dict[str, float]:
        return dict(self._breakdowns.get(seat_id, {}))

    def bag_to_seat(self, bag_bbox) -> str | None:
        """가방 박스 중심점이 속한 좌석 ID 반환. 없으면 None."""
        for seat in self._seats:
            if seat.region == (0, 0, 0, 0):
                continue
            if box_in_seat(bag_bbox, seat):
                return seat.seat_id
        return None
