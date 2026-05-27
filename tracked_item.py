"""TrackedItem — 추적 대상 물건의 상태 머신.

페이즈 6-A에서 TrackedBag → TrackedItem 리네임. 가방뿐 아니라 노트북·책 등
TRACKABLE_CLASSES로 확장된 모든 추적 물건을 동일 클래스로 다룬다.

각 추적 대상에 대한 상태·타이머·체류 기록을 캡슐화한다.
VLM 직접 호출은 하지 않고, 비동기 호출 결과(Future)와 컨텍스트만 보관한다
(메인 파이프라인이 결과를 소비).
"""
from concurrent.futures import Future

from config import (
    ST_TRACKING,
    T_LOST,
    PERSON_GRACE_SEC,
    SAFE_DISTANCE,
    DWELL_WINDOW_SEC,
    DWELL_OWNER_SEC,
)
from utils import dist, box_overlap_ratio


class TrackedItem:
    """하나의 추적 대상(가방/노트북/책 등)에 대한 상태·타이머·체류 기록을 캡슐화."""

    def __init__(self, master_id, center, size, start_time, cls=None):
        self.master_id       = master_id
        self.start_time      = start_time
        self.last_seen       = start_time
        self.center          = center
        self.anchor_center   = center
        self.size            = size
        self.state           = ST_TRACKING

        # YOLO 클래스 ID — 점유 가중치 룩업·VLM 컨텍스트에 사용.
        self.cls: int | None = cls

        # 체류 기록
        self.dwell_log: list[tuple[float, float]] = []
        self.person_near_since: float | None       = None
        self.person_last_detected: float           = 0.0   # 마지막으로 사람이 감지된 시각

        # 좌석 매핑 (등록 직후 SeatOccupancy.item_to_seat()으로 채움)
        self.seat_id: str | None = None

        # 페이즈 6-C: TRACKING → SUSPICIOUS 게이트 보류 플래그.
        # cafe+OCCUPIED로 보류 중일 때 True. elapsed는 계속 누적되며,
        # 좌석 점수가 OCCUPIED 미만으로 떨어지면 즉시 SUSPICIOUS 정상 진입.
        self.suspicious_held: bool = False

        # VLM
        self.vlm_called    = False
        self.last_vlm_call = 0.0
        self._last_reason  = ""
        self._last_image   = ""
        self._bbox         = (0, 0, 0, 0)
        self._vlm_future: Future | None = None   # 비동기 VLM 호출 결과
        self._vlm_context: dict | None  = None   # VLM 호출 시 컨텍스트
        self._last_cooldown_print: float = 0.0   # 쿨다운 메시지 중복 출력 방지

    # ── 헬퍼 ──────────────────────────────────────────────
    @property
    def elapsed(self):
        return self.last_seen - self.start_time

    @property
    def score(self):
        """0~100 위험도 점수"""
        return min(100, (self.elapsed / T_LOST) * 100)

    @property
    def person_is_near(self) -> bool:
        """최근 PERSON_GRACE_SEC 이내에 사람이 감지된 적 있는지"""
        return (self.last_seen - self.person_last_detected) < PERSON_GRACE_SEC

    @property
    def person_near_duration(self) -> float:
        """현재 사람이 근처에 연속으로 머문 시간 (초)"""
        if self.person_near_since is None:
            return 0.0
        return self.last_seen - self.person_near_since

    def reset_to_tracking(self, current_time):
        """타이머·상태를 초기화하여 TRACKING으로 복귀"""
        self.start_time      = current_time
        self.state           = ST_TRACKING
        self.dwell_log       = []
        self.person_near_since = None
        self.vlm_called      = False
        self.suspicious_held = False

    # ── 체류 기록 갱신 ────────────────────────────────────
    def update_dwell(self, person_boxes, item_center, item_box, current_time):
        """사람-물건 근접 여부 갱신. 중심 거리 OR 박스 겹침으로 판단."""
        person_is_near = False

        for px1, py1, px2, py2 in person_boxes:
            # 방법 1: 중심점 거리
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            if dist(item_center, (pcx, pcy)) < SAFE_DISTANCE:
                person_is_near = True
                break
            # 방법 2: 바운딩 박스 겹침 (사람 몸이 물건과 닿아 있으면)
            if box_overlap_ratio(item_box, (px1, py1, px2, py2)) > 0 or \
               box_overlap_ratio((px1, py1, px2, py2), item_box) > 0:
                person_is_near = True
                break

        if person_is_near:
            self.person_last_detected = current_time
            if self.person_near_since is None:
                self.person_near_since = current_time
            else:
                # 기존 구간 갱신
                if self.dwell_log and self.dwell_log[-1][0] == self.person_near_since:
                    self.dwell_log[-1] = (self.person_near_since, current_time)
                else:
                    self.dwell_log.append((self.person_near_since, current_time))
        else:
            if self.person_near_since is not None:
                self.dwell_log.append((self.person_near_since, current_time))
                self.person_near_since = None

        # 오래된 기록 정리
        cutoff = current_time - DWELL_WINDOW_SEC
        self.dwell_log = [(s, e) for s, e in self.dwell_log if e >= cutoff]

    # ── CV 기반 의심 판단 ─────────────────────────────────
    def check_suspicious(self, current_time) -> str:
        """
        "WARNING"  → 행인만 → VLM 없이 WARNING 직행
        "CALL_VLM" → 주인 후보 있음 → VLM 호출 필요
        """
        window_start = current_time - DWELL_WINDOW_SEC
        recent = [(s, e) for s, e in self.dwell_log if e >= window_start]

        if not recent:
            return "WARNING"

        max_dwell = max(
            (min(e, current_time) - max(s, window_start)) for s, e in recent
        )
        return "CALL_VLM" if max_dwell >= DWELL_OWNER_SEC else "WARNING"
