import cv2
import time
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from config import *
from utils import dist, box_overlap_ratio, apply_privacy_filter
from vlm_backend import call_vlm
from tracked_item import TrackedItem
from renderer import OverlayRenderer
from board import save_board_post, remove_board_post
from logger import save_log, save_alert
from seats import SEATS, box_in_seat
from seat_occupancy import SeatOccupancy
from tracker import ItemTracker


# ============================================================
#  성능 측정 — VLM 호출 시간 (별도 스레드에서 실행, 단발 출력)
# ============================================================
def _timed_call_vlm(*args, **kwargs):
    t0 = time.perf_counter()
    try:
        result = call_vlm(*args, **kwargs)
        return result
    finally:
        print(f"⏱️  VLM 호출: {(time.perf_counter() - t0) * 1000:.0f} ms")


# ============================================================
#  master_id monotonic 카운터 — BoxMOT track_id 재활용과 무관하게 유일성 보장
# ============================================================
_next_master_id = [1]   # 리스트로 mutable 참조

def get_next_master_id() -> int:
    mid = _next_master_id[0]
    _next_master_id[0] += 1
    return mid


# ============================================================
#  VLM 프롬프트용 좌석 컨텍스트 구성 — 카페 신호 + 점유 흔적 + 점수
# ============================================================
def _build_seat_info(seat_occupancy, seat_id, current_time) -> str:
    if not seat_id:
        return ""
    cafe   = seat_occupancy.is_cafe_active(seat_id, current_time)
    score  = seat_occupancy.get_score(seat_id)
    status = seat_occupancy.get_status(seat_id)
    breakdown = seat_occupancy.get_breakdown(seat_id)
    # 점유 흔적 리스트화 — 컵은 별도 라인으로 표시하므로 여기선 제외.
    bd_parts = []
    for name, val in breakdown.items():
        if name == "cup":
            continue
        if val > 0:
            bd_parts.append(f"{name}({val:.0f}점)")
    bd_str = ", ".join(bd_parts) if bd_parts else "없음"
    return (
        f"[좌석 {seat_id} 상황]\n"
        f"- 카페 이용 신호(컵): {'있음 ☕' if cafe else '없음'}\n"
        f"- 점유 흔적: {bd_str}\n"
        f"- 종합 점유 점수: {score:.0f}점 ({status})"
    )


# ============================================================
#  메인 파이프라인
# ============================================================
def main():
    # ── 초기화 ────────────────────────────────────────────
    yolo    = YOLO(YOLO_MODEL)
    tracker = ItemTracker()   # BoxMOT BotSort 어댑터 (DeepSort 인터페이스 호환)
    cap     = cv2.VideoCapture("case1.mp4")

    items: dict[int, TrackedItem] = {}   # track_id → TrackedItem
    last_gc       = 0.0
    frame_count   = 0
    renderer      = OverlayRenderer()
    vlm_executor  = ThreadPoolExecutor(max_workers=2)  # VLM 비동기 호출용
    pipeline_start = time.time()
    last_seat_debug = 0.0
    seat_occupancy = SeatOccupancy(SEATS)

    # ── 성능 측정 ─────────────────────────────────────────
    # 각 처리 단계별 시간을 deque로 누적 (최근 300개 = ~10초 at 30fps).
    stage_times: dict[str, deque] = {
        "frame_read":      deque(maxlen=300),
        "yolo":            deque(maxlen=300),
        "tracker":         deque(maxlen=300),
        "state_machine":   deque(maxlen=300),
        "seat_occupancy":  deque(maxlen=300),
        "renderer":        deque(maxlen=300),
        "total":           deque(maxlen=300),
    }
    last_perf_summary = time.perf_counter()
    print(f"⏱️  FRAME_SKIP={FRAME_SKIP} (실시간 기준: 처리 FPS × {FRAME_SKIP} ≥ 영상 FPS)")

    while cap.isOpened():
        iter_start = time.perf_counter()

        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_frame_read = time.perf_counter() - t0
        if not ret:
            break

        frame_count += 1
        if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
            continue

        current_time = time.time()

        # ── YOLO 추론 ────────────────────────────────────
        t0 = time.perf_counter()
        results    = yolo(frame, verbose=False, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)[0]
        t_yolo = time.perf_counter() - t0
        raw_item_detections = []
        person_boxes: list[tuple[int, int, int, int]] = []
        seat_indicators: list[tuple[int, int, int, int, int]] = []   # (x1,y1,x2,y2,cls)

        # 1차: 사람·추적 대상 물건·카페 컨텍스트 분리
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            w_box, h_box = x2 - x1, y2 - y1

            if cls == PERSON_CLASS:
                person_boxes.append((x1, y1, x2, y2))
            elif cls in TRACKABLE_CLASSES:
                # ── 크기/비율 필터링 ─────────────────────
                area = w_box * h_box
                if area < MIN_BOX_AREA:
                    continue
                aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
                if aspect > MAX_ASPECT_RATIO:
                    continue
                raw_item_detections.append(([x1, y1, w_box, h_box], conf, cls, (x1, y1, x2, y2)))
            elif cls in CAFE_CONTEXT_CLASSES:
                # 카페 컨텍스트 표식 (컵) — 추적 안 함, 매 프레임 raw로만 수집
                seat_indicators.append((x1, y1, x2, y2, cls))

        # 2차: 사람 박스와 크게 겹치는 물건 제거 (사람 몸을 물건으로 오인하는 문제)
        detections = []
        for det in raw_item_detections:
            item_box = det[3]  # (x1, y1, x2, y2)
            overlaps_person = False
            for pbox in person_boxes:
                if box_overlap_ratio(item_box, pbox) > PERSON_OVERLAP_THRESH:
                    overlaps_person = True
                    break
            if not overlaps_person:
                detections.append((det[0], det[1], det[2]))

        # ── 추적기 업데이트 (BoxMOT) ──────────────────────
        t0 = time.perf_counter()
        tracks = tracker.update_tracks(detections, frame=frame)
        t_tracker = time.perf_counter() - t0

        # ── 게시판 이미지용 깨끗한 프레임 (오버레이 전) ────
        clean_frame = frame.copy()

        # ── 트랙별 처리 ──────────────────────────────────
        t0 = time.perf_counter()
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            lx1, ly1, lx2, ly2 = map(int, track.to_ltrb())

            # ── 프레임 경계 클램프 (Kalman 필터 드리프트 방지)
            f_h, f_w = frame.shape[:2]
            x1 = max(0, lx1)
            y1 = max(0, ly1)
            x2 = min(f_w, lx2)
            y2 = min(f_h, ly2)
            w, h = x2 - x1, y2 - y1
            if w < 20 or h < 20:
                continue    # 너무 작으면 무시
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ── 신규 등록 / ID 상속 ──────────────────────
            if track_id not in items:
                inherited = False
                for old_id, old_item in list(items.items()):
                    age = current_time - old_item.last_seen
                    if age > INHERIT_TIME_WINDOW:
                        continue
                    if age < 0.5:
                        # 방금 갱신된 활성 트랙은 상속 후보 제외 (핑퐁 방지)
                        continue
                    if dist((cx, cy), old_item.center) < INHERIT_TOLERANCE:
                        # 상속: 구 항목 제거 후 새 ID로 등록
                        items[track_id] = old_item
                        items[track_id].last_seen = current_time
                        items[track_id].master_id = old_item.master_id
                        del items[old_id]
                        inherited = True
                        print(f"🔗 [ID 상속] master={old_item.master_id} ({old_id}→{track_id})")
                        break

                if not inherited:
                    det_cls = getattr(track, "det_class", None)
                    new_item = TrackedItem(
                        master_id=get_next_master_id(),
                        center=(cx, cy),
                        size=(w, h),
                        start_time=current_time,
                        cls=det_cls,
                    )
                    new_item._bbox = (x1, y1, x2, y2)
                    new_item.seat_id = seat_occupancy.item_to_seat(new_item._bbox)
                    items[track_id] = new_item
                    cls_label = SEAT_INDICATOR_NAMES.get(det_cls, str(det_cls))
                    print(f"🆕 [ID: {new_item.master_id}] 물건 등록 ({cls_label}) — "
                          f"seat={new_item.seat_id} bbox={new_item._bbox}")
                continue

            item = items[track_id]
            item.last_seen = current_time
            item._bbox     = (x1, y1, x2, y2)
            # 물건이 좌석 사이로 이동했을 가능성 — 매 프레임 재매핑
            item.seat_id   = seat_occupancy.item_to_seat(item._bbox)
            master_id      = item.master_id

            # ── 중심점 스무딩 (이동 판정 전에 먼저 수행) ────
            smoothed_cx = int(0.8 * item.center[0] + 0.2 * cx)
            smoothed_cy = int(0.8 * item.center[1] + 0.2 * cy)
            item.center = (smoothed_cx, smoothed_cy)

            # ── 물건 이동 감지 → 타이머 리셋 ─────────────
            # ※ 스무딩된 좌표로 비교 → 카메라 손떨림에 강인
            if dist(item.center, item.anchor_center) > MOVEMENT_TOLERANCE:
                print(f"🟢 [ID: {master_id}] 물건 이동 감지 → 타이머 리셋")
                item.reset_to_tracking(current_time)
                item.center        = (smoothed_cx, smoothed_cy)
                item.anchor_center = (smoothed_cx, smoothed_cy)
                item.size          = (w, h)
                continue

            # ── 앵커 드리프트 (감지 위치 흔들림 추적) ────────
            ax, ay = item.anchor_center
            item.anchor_center = (
                int(0.95 * ax + 0.05 * smoothed_cx),
                int(0.95 * ay + 0.05 * smoothed_cy),
            )

            # ── 체류 기록 갱신 ────────────────────────────
            item.update_dwell(person_boxes, (cx, cy), (x1, y1, x2, y2), current_time)

            # ── 상태 전이 ─────────────────────────────────
            state   = item.state

            # ── 사람이 근처에 있으면 방치 타이머 리셋 ────────
            # "방치 시간"은 사람이 떠난 시점부터 세야 함
            if state == ST_TRACKING and item.person_is_near:
                item.start_time = current_time

            elapsed = item.elapsed

            # ── 비동기 VLM 결과 수신 ─────────────────────────
            if item._vlm_future is not None and item._vlm_future.done():
                # VLM 호출 이후 물건이 이동하여 TRACKING으로 리셋된 경우 결과 버림
                if item.state == ST_TRACKING:
                    item._vlm_future = None
                    item._vlm_context = None
                    print(f"⚠️  [ID: {master_id}] VLM 결과 도착했으나 이미 TRACKING 리셋 → 결과 무시")
                else:
                    ctx = item._vlm_context or {}
                    try:
                        raw = item._vlm_future.result()
                        result_json = json.loads(raw)

                        stage = ctx.get("stage", "SUSPICIOUS")
                        print(f"{'=' * 40}")
                        print(f"🎯 [VLM - {stage}] {result_json['status']}")
                        print(f"   근거: {result_json['reason']}")
                        print(f"{'=' * 40}\n")

                        if stage == "SUSPICIOUS":
                            if result_json["status"] == "SAFE":
                                print(f"🟢 [ID: {master_id}] 주인 확인 → TRACKING 복귀")
                                item.reset_to_tracking(current_time)
                                state = ST_TRACKING
                            else:
                                item.state = ST_WARNING
                                state = ST_WARNING
                                item._last_reason = result_json.get("reason", "")
                                save_alert(master_id, ST_WARNING, result_json, item.score, LOCATION,
                                           ctx.get("fname", ""))
                            save_log(ctx.get("fname", ""), elapsed / 60, LOCATION,
                                     ST_SUSPICIOUS, result_json, VLM_BACKEND)

                        elif stage == "LOST":
                            if result_json.get("status") == "WARNING":
                                # VLM이 주인 발견 → TRACKING 복귀
                                print(f"🟢 [ID: {master_id}] 주인 확인 (LOST 재확인) → TRACKING 복귀")
                                item.reset_to_tracking(current_time)
                                state = ST_TRACKING
                                remove_board_post(master_id)
                            else:
                                item.state = ST_LOST
                                state = ST_LOST
                                item._last_reason = result_json.get("reason", "")
                                save_alert(master_id, ST_LOST, result_json, item.score, LOCATION,
                                           ctx.get("fname", ""))
                            save_log(ctx.get("fname", ""), elapsed / 60, LOCATION,
                                     ST_LOST, result_json, VLM_BACKEND)

                    except Exception as e:
                        stage = ctx.get("stage", "?")
                        print(f"❌ VLM 결과 처리 실패 ({stage}): {e}")
                        if stage == "SUSPICIOUS":
                            item.state = ST_WARNING
                        else:
                            item.state = ST_LOST

                    item._vlm_future = None
                    item._vlm_context = None

            # TRACKING → SUSPICIOUS (카페 컨텍스트 + 점유 흔적 게이트)
            if state == ST_TRACKING and item.person_near_duration < PASSERBY_MAX_SEC:
                seat_id     = item.seat_id
                cafe_active = seat_occupancy.is_cafe_active(seat_id, current_time)
                threshold   = T_TRACKING_CAFE if cafe_active else T_TRACKING_NOCAFE

                if elapsed >= threshold:
                    seat_st = seat_occupancy.get_status(seat_id) if seat_id else "ABANDONED"
                    seat_sc = seat_occupancy.get_score(seat_id) if seat_id else 0

                    if cafe_active and seat_st == "OCCUPIED":
                        # 컵 있음 + 점유 흔적 충분 → 잠깐 자리 비움. 한 번 더 기회.
                        item.start_time = current_time
                        print(f"💺 [ID: {master_id}] 좌석 {seat_id} 카페 이용 중"
                              f"(점유 {seat_sc:.0f}점 {seat_st}) → SUSPICIOUS 보류")
                    else:
                        cafe_tag = "☕ 카페" if cafe_active else "🚪 카페外"
                        print(f"\n⏱️  [ID: {master_id}] {elapsed:.0f}초 경과 {cafe_tag} "
                              f"(점유 {seat_sc:.0f}점 {seat_st}) → SUSPICIOUS")
                        item.state = ST_SUSPICIOUS
                        state      = ST_SUSPICIOUS

            # SUSPICIOUS 판단
            if state == ST_SUSPICIOUS and not item.vlm_called:
                cv_result = item.check_suspicious(current_time)

                if cv_result == "WARNING":
                    # 행인만 → VLM 없이 WARNING 직행
                    print(f"📊 [ID: {master_id}] CV 판단: 행인만 → WARNING 직행")
                    item.state      = ST_WARNING
                    item.vlm_called = True
                    state           = ST_WARNING

                    safe_img = apply_privacy_filter(frame, person_boxes, (x1, y1, x2, y2))
                    if safe_img is not None:
                        fname = f"trigger_event_{master_id}.jpg"
                        cv2.imwrite(fname, safe_img)
                        result_json = {
                            "status": "WARNING",
                            "reason": f"CV 판단: 최근 {DWELL_WINDOW_SEC}초 내 {DWELL_OWNER_SEC}초 이상 머문 사람 없음",
                        }
                        item._last_reason = result_json["reason"]
                        save_alert(master_id, ST_WARNING, result_json, item.score, LOCATION, fname)
                        save_log(fname, elapsed / 60, LOCATION, ST_WARNING, result_json, "CV")
                        print(f"🟡 [ID: {master_id}] WARNING — {result_json['reason']}")

                else:
                    # 주인 후보 → VLM 비동기 호출
                    if current_time - item.last_vlm_call >= API_COOLDOWN:
                        print(f"🤖 [ID: {master_id}] CV: 주인 후보 감지 → VLM 비동기 호출")
                        safe_img = apply_privacy_filter(frame, person_boxes, (x1, y1, x2, y2))
                        if safe_img is not None:
                            fname = f"trigger_event_{master_id}.jpg"
                            cv2.imwrite(fname, safe_img)
                            item.last_vlm_call = current_time
                            item.vlm_called    = True

                            question = (
                                "이미지 속 물건 주변에 10초 이상 머물고 있는 사람이 보이나요? "
                                "그 사람이 물건의 주인처럼 보인다면 SAFE, "
                                "그냥 지나치는 행인이거나 아무도 없다면 WARNING으로 판정하세요."
                            )
                            seat_info = _build_seat_info(seat_occupancy, item.seat_id, current_time)
                            item._vlm_future = vlm_executor.submit(
                                _timed_call_vlm, fname, elapsed / 60, LOCATION, question,
                                "SAFE 또는 WARNING", seat_info
                            )
                            item._vlm_context = {"stage": "SUSPICIOUS", "fname": fname}
                    else:
                        if current_time - item._last_cooldown_print > 1.0:
                            remaining = API_COOLDOWN - (current_time - item.last_vlm_call)
                            print(f"⏳ [ID: {master_id}] VLM 쿨다운 ({remaining:.0f}초)")
                            item._last_cooldown_print = current_time

            # WARNING/LOST 중 주인 복귀 감지 (엄격한 기준)
            if state in (ST_WARNING, ST_LOST) and item.person_near_duration >= DWELL_RETURN_SEC:
                print(f"👀 [ID: {master_id}] {state} 중 사람 {DWELL_RETURN_SEC}초 이상 체류 → TRACKING")
                item.reset_to_tracking(current_time)
                state = ST_TRACKING
                result_json = {"status": "SAFE", "reason": f"사람이 {DWELL_RETURN_SEC}초 이상 연속 체류하여 주인 복귀로 판단"}
                item._last_reason = result_json["reason"]
                save_log("none.jpg", elapsed / 60, LOCATION, ST_TRACKING, result_json, "CV")
                remove_board_post(master_id)  # 게시판에서도 삭제

            # WARNING → LOST (카페 컨텍스트 + 점유 흔적 게이트)
            # 페이즈 6-B: SUSPICIOUS 게이트와 동일한 비대칭 처리를 LOST에도 적용.
            # 카페+OCCUPIED는 T_LOST 직후 60초간 LOST 호출 보류 → 점유 신호가
            # 끊기거나 시간 충분히 더 지나야 VLM 호출.
            if state == ST_WARNING and elapsed >= T_LOST:
                seat_id     = item.seat_id
                cafe_active = seat_occupancy.is_cafe_active(seat_id, current_time) if seat_id else False
                seat_st     = seat_occupancy.get_status(seat_id) if seat_id else "ABANDONED"
                seat_sc     = seat_occupancy.get_score(seat_id) if seat_id else 0
                cafe_tag    = "☕ 카페" if cafe_active else "🚪 카페外"

                if cafe_active and seat_st == "OCCUPIED" and elapsed < T_LOST + 60:
                    # 점유 흔적 유지 → 추가 60초 유예. 로그 throttling.
                    if frame_count % 30 == 0:
                        print(f"💺 [ID: {master_id}] 좌석 {seat_id} 카페 OCCUPIED "
                              f"({seat_sc:.0f}점) → LOST 호출 보류 "
                              f"({elapsed:.0f}/{T_LOST + 60}초)")
                else:
                    if item._vlm_future is None and current_time - item.last_vlm_call >= API_COOLDOWN:
                        print(f"\n🚨 [ID: {master_id}] {elapsed:.0f}초 경과 {cafe_tag} "
                              f"(점유 {seat_sc:.0f}점 {seat_st}) → LOST VLM 호출")
                        safe_img = apply_privacy_filter(frame, person_boxes, (x1, y1, x2, y2))
                        if safe_img is not None:
                            fname = f"trigger_event_{master_id}_lost.jpg"
                            cv2.imwrite(fname, safe_img)
                            item.last_vlm_call = current_time

                            question = (
                                f"이 물건은 약 {elapsed:.0f}초 동안 같은 자리에 있습니다.\n"
                                f"좌석 점유 흔적과 카페 컨텍스트를 함께 고려해서:\n"
                                f"- 컵·다른 소지품이 남아 있고 잠깐 비웠을 가능성이 보이면 WARNING\n"
                                f"- 좌석이 완전히 빈 상태이거나 주인 복귀 흔적이 없으면 DANGER\n"
                                f"으로 판정하세요."
                            )
                            seat_info = _build_seat_info(seat_occupancy, seat_id, current_time)
                            item._vlm_future = vlm_executor.submit(
                                _timed_call_vlm, fname, elapsed / 60, LOCATION, question,
                                "WARNING 또는 DANGER", seat_info
                            )
                            item._vlm_context = {"stage": "LOST", "fname": fname}

            # ── 오버레이 그리기 ───────────────────────────
            renderer.draw_safe_radius(frame, cx, cy)
            renderer.draw_item(frame, x1, y1, x2, y2, item)
        t_state = time.perf_counter() - t0

        # ── 좌석 점유 점수 + 카페 컨텍스트 갱신 ───────────
        # 활성 물건의 최신 _bbox가 반영된 시점에 호출.
        t0 = time.perf_counter()
        seat_occupancy.update(person_boxes, items, seat_indicators, current_time)
        t_seat = time.perf_counter() - t0

        # ── 렌더링 + 보드 + 디버그 + 표시 ─────────────────
        t_render_start = time.perf_counter()

        # ── 사람 바운딩 박스 그리기 ───────────────────────
        for px1, py1, px2, py2 in person_boxes:
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            is_near = any(
                dist((pcx, pcy), it.center) < SAFE_DISTANCE
                for it in items.values()
            )
            renderer.draw_person(frame, px1, py1, px2, py2, is_near)

        # ── 디버그 오버레이: 좌석 영역 + 좌석 표식 박스 ─────
        if DEBUG_DRAW_DETECTIONS:
            # 좌석 영역 — 점유 상태별 색 (OCCUPIED 초록 / PARTIAL 노랑 / ABANDONED 회색)
            renderer.draw_seats(frame, SEATS, seat_occupancy)

            # 좌석 표식 박스 (회색 + 클래스명) — 추적되지 않는 표식만 (현재 컵).
            for ix1, iy1, ix2, iy2, cls in seat_indicators:
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (180, 180, 180), 1)
                label = SEAT_INDICATOR_NAMES.get(cls, str(cls))
                cv2.putText(frame, label, (ix1, iy1 - 3),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (180, 180, 180), 1, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────
        renderer.draw_hud(frame, len(items), len(person_boxes))

        # ── 게시판 + 대시보드 업데이트 ─────────────────────
        for it in items.values():
            if STATE_PRIORITY.get(it.state, 0) >= STATE_PRIORITY[ST_WARNING]:
                web_status_map = {ST_WARNING: "WARNING", ST_SUSPICIOUS: "WARNING", ST_LOST: "DANGER"}
                web_status = web_status_map.get(it.state, "WARNING")
                reason = getattr(it, '_last_reason', f"상태: {it.state}")

                # 바운딩 박스 클램프
                f_h, f_w = frame.shape[:2]
                bx1 = max(0, it._bbox[0])
                by1 = max(0, it._bbox[1])
                bx2 = min(f_w, it._bbox[2])
                by2 = min(f_h, it._bbox[3])

                if bx2 > bx1 + 10 and by2 > by1 + 10:
                    snoozed = save_board_post(
                        master_id=it.master_id,
                        status=web_status,
                        reason=reason,
                        score=it.score,
                        location=LOCATION,
                        image_frame=clean_frame,
                        person_boxes=person_boxes,
                        bag_box=(bx1, by1, bx2, by2),
                    )
                    if snoozed:
                        it.reset_to_tracking(current_time)

        # ── 좌석 점유 디버그 (5초마다) ────────────────────
        if current_time - last_seat_debug >= 5.0:
            elapsed_sec = int(current_time - pipeline_start)
            # 전역 카운트는 추적되지 않는 raw 표식(현재 컵)만 표시.
            global_counts = {name: 0 for name in SEAT_INDICATOR_NAMES.values()}
            for _, _, _, _, cls in seat_indicators:
                global_counts[SEAT_INDICATOR_NAMES.get(cls, str(cls))] = \
                    global_counts.get(SEAT_INDICATOR_NAMES.get(cls, str(cls)), 0) + 1
            item_count = len(items)
            counts_str = " ".join(f"{k}={v}" for k, v in global_counts.items() if v)
            print(f"[t={elapsed_sec}s] global: items={item_count} {counts_str}")
            for seat in SEATS:
                if seat.region == (0, 0, 0, 0):
                    print(f"  seat_{seat.seat_id}: (영역 미설정)")
                    continue
                item_in = sum(1 for it in items.values() if it.seat_id == seat.seat_id)
                ind_in = {name: 0 for name in SEAT_INDICATOR_NAMES.values()}
                for x1, y1, x2, y2, cls in seat_indicators:
                    if box_in_seat((x1, y1, x2, y2), seat):
                        nm = SEAT_INDICATOR_NAMES.get(cls, str(cls))
                        ind_in[nm] = ind_in.get(nm, 0) + 1
                ind_str = " ".join(f"{k}={v}" for k, v in ind_in.items() if v)
                score  = seat_occupancy.get_score(seat.seat_id)
                raw_old, raw_new = seat_occupancy.get_score_raw_range(seat.seat_id)
                status = seat_occupancy.get_status(seat.seat_id)
                cafe   = "○" if seat_occupancy.is_cafe_active(seat.seat_id, current_time) else "✗"
                print(f"  seat_{seat.seat_id}: cafe={cafe} score={int(score)} "
                      f"(raw {int(raw_old)}→{int(raw_new)}) [{status}] "
                      f"items={item_in} {ind_str}")
            if items:
                item_rows = ", ".join(
                    f"ID={it.master_id} cls={it.cls} seat={it.seat_id or '-'} state={it.state}"
                    for it in items.values()
                )
                print(f"  items: {item_rows}")
            last_seat_debug = current_time

        # ── GC: 오래된 객체 제거 ──────────────────────────
        if current_time - last_gc > GC_INTERVAL:
            before = len(items)
            items = {
                tid: it for tid, it in items.items()
                if current_time - it.last_seen < GC_MAX_AGE
            }
            removed = before - len(items)
            if removed:
                print(f"🗑️  GC: {removed}개 오래된 객체 제거")
            last_gc = current_time

        # ── 화면 출력 (고해상도 영상 리사이즈) ────────────
        display = frame
        if DISPLAY_WIDTH and frame.shape[1] > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / frame.shape[1]
            display = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow("Intelligent Surveillance AI", display)
        t_renderer = time.perf_counter() - t_render_start

        # ── 단계별 시간 누적 + 30초마다 요약 출력 ─────────
        stage_times["frame_read"].append(t_frame_read)
        stage_times["yolo"].append(t_yolo)
        stage_times["tracker"].append(t_tracker)
        stage_times["state_machine"].append(t_state)
        stage_times["seat_occupancy"].append(t_seat)
        stage_times["renderer"].append(t_renderer)
        stage_times["total"].append(time.perf_counter() - iter_start)

        if time.perf_counter() - last_perf_summary >= 30.0:
            last_perf_summary = time.perf_counter()
            print("⏱️  단계별 처리 시간 (최근 평균, ms):")
            for stage, times in stage_times.items():
                if times:
                    avg_ms = sum(times) / len(times) * 1000
                    print(f"   {stage:<15} {avg_ms:>6.1f} ms")
            if stage_times["total"]:
                total_avg = sum(stage_times["total"]) / len(stage_times["total"])
                fps_real = 1.0 / total_avg if total_avg > 0 else 0
                print(f"   실제 처리 FPS: {fps_real:.1f}  "
                      f"(실시간 기준: ≥ 영상 FPS / {FRAME_SKIP})")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            # space: 일시정지 — 다시 space로 재개, q로 종료
            while True:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 == ord(" "):
                    break
                if k2 == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    vlm_executor.shutdown(wait=False)
                    return

    cap.release()
    cv2.destroyAllWindows()
    vlm_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
