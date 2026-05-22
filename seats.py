"""좌석 데이터 모델 — 좌석 영역 정의 및 박스-좌석 포함 판단.

SEATS의 region 좌표는 영상별로 달라지므로, 영상 확인 후 조정해야 한다.
(0, 0, 0, 0)으로 두면 모든 박스가 어느 좌석에도 속하지 않는다 — 의도된 초기값.
"""
from dataclasses import dataclass


@dataclass
class Seat:
    seat_id: str
    region: tuple[int, int, int, int]   # (x1, y1, x2, y2)


# TODO: 영상 보고 좌표 조정
SEATS: list[Seat] = [
    Seat("A", (233, 265, 1001, 717)),
    Seat("B", (3, 240, 207, 568)),
    Seat("C", (0, 0, 0, 0)),
]


def box_in_seat(bbox, seat: Seat) -> bool:
    """박스 중심점이 좌석 영역 안에 있는지 판단. bbox = (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sx1, sy1, sx2, sy2 = seat.region
    return sx1 <= cx <= sx2 and sy1 <= cy <= sy2
