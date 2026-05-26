"""ItemTracker — BoxMOT BotSort를 DeepSort 인터페이스로 래핑한 어댑터.

DeepSort(`deep_sort_realtime`)의 `update_tracks()` 호출 형식을 그대로 유지하여,
메인 파이프라인에서는 tracker 객체 생성 한 줄만 교체하면 된다.

페이즈 5에서 도입, 페이즈 6-A에서 BagTracker → ItemTracker로 리네임
(가방·노트북·책 등 TRACKABLE_CLASSES 모두 추적).
ReID 임베딩은 BoxMOT 내부에서만 사용 (외부 노출 X — 페이즈 6에서 처리).
"""
from __future__ import annotations

import numpy as np
import torch

from boxmot.reid import ReID
from boxmot.trackers.botsort.botsort import BotSort


class _AdaptedTrack:
    """BoxMOT TrackResults 한 행을 DeepSort Track 인터페이스로 노출."""

    def __init__(self, track_id: int, ltrb: list[float], det_class: int | None = None):
        self.track_id = int(track_id)
        self._ltrb = ltrb
        # DeepSort 호환 — `track.det_class`로 YOLO 클래스 ID 노출.
        self.det_class: int | None = det_class

    def is_confirmed(self) -> bool:
        # BoxMOT은 confirmed 트랙만 반환하므로 항상 True.
        return True

    def to_ltrb(self) -> list[float]:
        return self._ltrb


class ItemTracker:
    """DeepSort-compatible 다중 물건 추적기 (내부 구현: BoxMOT BotSort).

    Parameters
    ----------
    reid_weights : str
        ReID 모델 파일명. BoxMOT이 첫 호출 시 자동 다운로드.
        기본값 osnet_x0_25_msmt17.pt — 가벼운 모델로 충분.
    """

    def __init__(self, reid_weights: str = "osnet_x0_25_msmt17.pt", use_reid: bool = False):
        # 페이즈 6-B 속도 최적화: ReID 비활성화가 기본값.
        # tracker 단계 76ms → IoU만으로 충분 (책상 위 정적 물건 시나리오).
        # 가방·노트북이 잠깐 가려졌다 나올 때 ID가 바뀔 수 있지만, 좌석 매핑이
        # 핵심이라 시스템 동작에 큰 영향 없음.
        # 사람 ReID는 페이즈 7에서 별도 처리 예정.
        reid_backend = None
        if use_reid:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            reid_backend = ReID(weights=reid_weights, device=device).model
        # max_age=150 / min_hits=3 → 기존 DeepSort(max_age=150, n_init=3)와 동등.
        # track_buffer=150 → BotSort 내부 lost-track 보관 기간을 max_age에 맞춤.
        self._tracker = BotSort(
            reid_model=reid_backend,
            max_age=150,
            min_hits=3,
            track_buffer=150,
            with_reid=use_reid,
        )

    # ── DeepSort 호환 메서드 ────────────────────────────────
    def update_tracks(self, raw_detections, frame=None):
        """DeepSort.update_tracks와 호환되는 시그니처.

        Parameters
        ----------
        raw_detections : list of ([x1, y1, w, h], conf, cls)
            DeepSort 형식. bbox는 LTWH.
        frame : np.ndarray (BGR)
            BoxMOT은 ReID 특징 추출을 위해 프레임이 반드시 필요.

        Returns
        -------
        list[_AdaptedTrack]
            각 트랙은 .is_confirmed(), .track_id, .to_ltrb(), .det_class를 제공.
        """
        if not raw_detections:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            rows = []
            for ltwh, conf, cls in raw_detections:
                x1, y1, w, h = ltwh
                rows.append([x1, y1, x1 + w, y1 + h, float(conf), float(cls)])
            dets = np.array(rows, dtype=np.float32)

        if frame is None:
            raise ValueError("ItemTracker.update_tracks requires `frame=` (BoxMOT 요구사항)")

        results = self._tracker.update(dets, frame)
        arr = np.asarray(results)

        out: list[_AdaptedTrack] = []
        # arr 행 형식: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        for row in arr:
            tid = row[4]
            cls = int(row[6])
            ltrb = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            out.append(_AdaptedTrack(int(tid), ltrb, det_class=cls))
        return out
