import cv2
import time
import json
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import *
from utils import dist, box_overlap_ratio, apply_privacy_filter
from vlm_backend import call_vlm
from tracked_bag import TrackedBag
from renderer import OverlayRenderer
from board import save_board_post, remove_board_post
from logger import save_log, save_alert
from seats import SEATS, box_in_seat


# ============================================================
#  메인 파이프라인
# ============================================================
def main():
    # ── 초기화 ────────────────────────────────────────────
    yolo    = YOLO(YOLO_MODEL)
    tracker = DeepSort(max_age=150, n_init=3, max_iou_distance=0.9)
    cap     = cv2.VideoCapture("case1.mp4")

    LOCATION = "캠퍼스 열람실"   # 고정값 (운영 시에는 카메라별로 다르게 설정)

    bags: dict[int, TrackedBag] = {}   # track_id → TrackedBag
    last_gc       = 0.0
    frame_count   = 0
    renderer      = OverlayRenderer()
    vlm_executor  = ThreadPoolExecutor(max_workers=2)  # VLM 비동기 호출용
    pipeline_start = time.time()
    last_seat_debug = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
            continue

        current_time = time.time()

        # ── YOLO 추론 ────────────────────────────────────
        results    = yolo(frame, verbose=False, imgsz=736, conf=YOLO_CONF)[0]
        raw_bag_detections = []
        person_boxes: list[tuple[int, int, int, int]] = []
        seat_indicators: list[tuple[int, int, int, int, int]] = []   # (x1,y1,x2,y2,cls)

        # 1차: 사람과 가방 후보 분리
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            w_box, h_box = x2 - x1, y2 - y1

            if cls == PERSON_CLASS:
                person_boxes.append((x1, y1, x2, y2))
            elif cls in TARGET_CLASSES:
                # ── 크기/비율 필터링 ─────────────────────
                area = w_box * h_box
                if area < MIN_BOX_AREA:
                    continue
                aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
                if aspect > MAX_ASPECT_RATIO:
                    continue
                raw_bag_detections.append(([x1, y1, w_box, h_box], conf, cls, (x1, y1, x2, y2)))
            elif cls in SEAT_INDICATOR_CLASSES:
                # 좌석 점유 표식 — 필터링 없이 매 프레임 새로 수집 (추적 X)
                seat_indicators.append((x1, y1, x2, y2, cls))

        # 2차: 사람 박스와 크게 겹치는 가방 제거 (사람 몸을 가방으로 오인하는 문제)
        detections = []
        for det in raw_bag_detections:
            bag_box = det[3]  # (x1, y1, x2, y2)
            overlaps_person = False
            for pbox in person_boxes:
                if box_overlap_ratio(bag_box, pbox) > PERSON_OVERLAP_THRESH:
                    overlaps_person = True
                    break
            if not overlaps_person:
                detections.append((det[0], det[1], det[2]))

        # ── DeepSORT 업데이트 ─────────────────────────────
        tracks = tracker.update_tracks(detections, frame=frame)

        # ── 게시판 이미지용 깨끗한 프레임 (오버레이 전) ────
        clean_frame = frame.copy()

        # ── 트랙별 처리 ──────────────────────────────────
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            lx1, ly1, lx2, ly2 = map(int, track.to_ltrb())

            # ── 프레임 경계 클램프 (DeepSORT 칼만 필터 드리프트 방지)
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
            if track_id not in bags:
                inherited = False
                for old_id, old_bag in list(bags.items()):
                    age = current_time - old_bag.last_seen
                    if age > INHERIT_TIME_WINDOW:
                        continue
                    if age < 0.5:
                        # 방금 갱신된 활성 트랙은 상속 후보 제외 (핑퐁 방지)
                        continue
                    if dist((cx, cy), old_bag.center) < INHERIT_TOLERANCE:
                        # 상속: 구 항목 제거 후 새 ID로 등록
                        bags[track_id] = old_bag
                        bags[track_id].last_seen = current_time
                        bags[track_id].master_id = old_bag.master_id
                        del bags[old_id]
                        inherited = True
                        print(f"🔗 [ID 상속] master={old_bag.master_id} ({old_id}→{track_id})")
                        break

                if not inherited:
                    bags[track_id] = TrackedBag(
                        master_id=track_id,
                        center=(cx, cy),
                        size=(w, h),
                        start_time=current_time,
                    )
                continue

            bag = bags[track_id]
            bag.last_seen = current_time
            bag._bbox     = (x1, y1, x2, y2)
            master_id     = bag.master_id

            # ── 중심점 스무딩 (이동 판정 전에 먼저 수행) ────
            smoothed_cx = int(0.8 * bag.center[0] + 0.2 * cx)
            smoothed_cy = int(0.8 * bag.center[1] + 0.2 * cy)
            bag.center = (smoothed_cx, smoothed_cy)

            # ── 가방 이동 감지 → 타이머 리셋 ─────────────
            # ※ 스무딩된 좌표로 비교 → 카메라 손떨림에 강인
            if dist(bag.center, bag.anchor_center) > MOVEMENT_TOLERANCE:
                print(f"🟢 [ID: {master_id}] 가방 이동 감지 → 타이머 리셋")
                bag.reset_to_tracking(current_time)
                bag.center        = (smoothed_cx, smoothed_cy)
                bag.anchor_center = (smoothed_cx, smoothed_cy)
                bag.size          = (w, h)
                continue

            # ── 앵커 드리프트 (감지 위치 흔들림 추적) ────────
            ax, ay = bag.anchor_center
            bag.anchor_center = (
                int(0.95 * ax + 0.05 * smoothed_cx),
                int(0.95 * ay + 0.05 * smoothed_cy),
            )

            # ── 체류 기록 갱신 ────────────────────────────
            bag.update_dwell(person_boxes, (cx, cy), (x1, y1, x2, y2), current_time)

            # ── 상태 전이 ─────────────────────────────────
            state   = bag.state

            # ── 사람이 근처에 있으면 방치 타이머 리셋 ────────
            # "방치 시간"은 사람이 떠난 시점부터 세야 함
            if state == ST_TRACKING and bag.person_is_near:
                bag.start_time = current_time

            elapsed = bag.elapsed

            # ── 비동기 VLM 결과 수신 ─────────────────────────
            if bag._vlm_future is not None and bag._vlm_future.done():
                # VLM 호출 이후 가방이 이동하여 TRACKING으로 리셋된 경우 결과 버림
                if bag.state == ST_TRACKING:
                    bag._vlm_future = None
                    bag._vlm_context = None
                    print(f"⚠️  [ID: {master_id}] VLM 결과 도착했으나 이미 TRACKING 리셋 → 결과 무시")
                else:
                    ctx = bag._vlm_context or {}
                    try:
                        raw = bag._vlm_future.result()
                        result_json = json.loads(raw)

                        stage = ctx.get("stage", "SUSPICIOUS")
                        print(f"{'=' * 40}")
                        print(f"🎯 [VLM - {stage}] {result_json['status']}")
                        print(f"   근거: {result_json['reason']}")
                        print(f"{'=' * 40}\n")

                        if stage == "SUSPICIOUS":
                            if result_json["status"] == "SAFE":
                                print(f"🟢 [ID: {master_id}] 주인 확인 → TRACKING 복귀")
                                bag.reset_to_tracking(current_time)
                                state = ST_TRACKING
                            else:
                                bag.state = ST_WARNING
                                state = ST_WARNING
                                bag._last_reason = result_json.get("reason", "")
                                save_alert(master_id, ST_WARNING, result_json, bag.score, LOCATION,
                                           ctx.get("fname", ""))
                            save_log(ctx.get("fname", ""), elapsed / 60, LOCATION,
                                     ST_SUSPICIOUS, result_json, VLM_BACKEND)

                        elif stage == "LOST":
                            if result_json.get("status") == "WARNING":
                                # VLM이 주인 발견 → TRACKING 복귀
                                print(f"🟢 [ID: {master_id}] 주인 확인 (LOST 재확인) → TRACKING 복귀")
                                bag.reset_to_tracking(current_time)
                                state = ST_TRACKING
                                remove_board_post(master_id)
                            else:
                                bag.state = ST_LOST
                                state = ST_LOST
                                bag._last_reason = result_json.get("reason", "")
                                save_alert(master_id, ST_LOST, result_json, bag.score, LOCATION,
                                           ctx.get("fname", ""))
                            save_log(ctx.get("fname", ""), elapsed / 60, LOCATION,
                                     ST_LOST, result_json, VLM_BACKEND)

                    except Exception as e:
                        stage = ctx.get("stage", "?")
                        print(f"❌ VLM 결과 처리 실패 ({stage}): {e}")
                        if stage == "SUSPICIOUS":
                            bag.state = ST_WARNING
                        else:
                            bag.state = ST_LOST

                    bag._vlm_future = None
                    bag._vlm_context = None

            # TRACKING → SUSPICIOUS
            if state == ST_TRACKING and elapsed >= T_TRACKING:
                print(f"\n⏱️  [ID: {master_id}] {T_TRACKING}초 경과 + 사람 없음 → SUSPICIOUS")
                bag.state = ST_SUSPICIOUS
                state     = ST_SUSPICIOUS

            # SUSPICIOUS 판단
            if state == ST_SUSPICIOUS and not bag.vlm_called:
                cv_result = bag.check_suspicious(current_time)

                if cv_result == "WARNING":
                    # 행인만 → VLM 없이 WARNING 직행
                    print(f"📊 [ID: {master_id}] CV 판단: 행인만 → WARNING 직행")
                    bag.state      = ST_WARNING
                    bag.vlm_called = True
                    state          = ST_WARNING

                    safe_img = apply_privacy_filter(frame, person_boxes, (x1, y1, x2, y2))
                    if safe_img is not None:
                        fname = f"trigger_event_{master_id}.jpg"
                        cv2.imwrite(fname, safe_img)
                        result_json = {
                            "status": "WARNING",
                            "reason": f"CV 판단: 최근 {DWELL_WINDOW_SEC}초 내 {DWELL_OWNER_SEC}초 이상 머문 사람 없음",
                        }
                        bag._last_reason = result_json["reason"]
                        save_alert(master_id, ST_WARNING, result_json, bag.score, LOCATION, fname)
                        save_log(fname, elapsed / 60, LOCATION, ST_WARNING, result_json, "CV")
                        print(f"🟡 [ID: {master_id}] WARNING — {result_json['reason']}")

                else:
                    # 주인 후보 → VLM 비동기 호출
                    if current_time - bag.last_vlm_call >= API_COOLDOWN:
                        print(f"🤖 [ID: {master_id}] CV: 주인 후보 감지 → VLM 비동기 호출")
                        safe_img = apply_privacy_filter(frame, person_boxes, (x1, y1, x2, y2))
                        if safe_img is not None:
                            fname = f"trigger_event_{master_id}.jpg"
                            cv2.imwrite(fname, safe_img)
                            bag.last_vlm_call = current_time
                            bag.vlm_called    = True

                            question = (
                                "이미지 속 가방 주변에 10초 이상 머물고 있는 사람이 보이나요? "
                                "그 사람이 가방의 주인처럼 보인다면 SAFE, "
                                "그냥 지나치는 행인이거나 아무도 없다면 WARNING으로 판정하세요."
                            )
                            bag._vlm_future = vlm_executor.submit(
                                call_vlm, fname, elapsed / 60, LOCATION, question,
                                "SAFE 또는 WARNING"
                            )
                            bag._vlm_context = {"stage": "SUSPICIOUS", "fname": fname}
                    else:
                        if current_time - bag._last_cooldown_print > 1.0:
                            remaining = API_COOLDOWN - (current_time - bag.last_vlm_call)
                            print(f"⏳ [ID: {master_id}] VLM 쿨다운 ({remaining:.0f}초)")
                            bag._last_cooldown_print = current_time

            # WARNING/LOST 중 주인 복귀 감지 (엄격한 기준)
            if state in (ST_WARNING, ST_LOST) and bag.person_near_duration >= DWELL_RETURN_SEC:
                print(f"👀 [ID: {master_id}] {state} 중 사람 {DWELL_RETURN_SEC}초 이상 체류 → TRACKING")
                bag.reset_to_tracking(current_time)
                state = ST_TRACKING
                result_json = {"status": "SAFE", "reason": f"사람이 {DWELL_RETURN_SEC}초 이상 연속 체류하여 주인 복귀로 판단"}
                bag._last_reason = result_json["reason"]
                save_log("none.jpg", elapsed / 60, LOCATION, ST_TRACKING, result_json, "CV")
                remove_board_post(master_id)  # 게시판에서도 삭제

            # WARNING → LOST (VLM 2차 비동기)
            if state == ST_WARNING and elapsed >= T_LOST:
                if bag._vlm_future is None and current_time - bag.last_vlm_call >= API_COOLDOWN:
                    print(f"\n🚨 [ID: {master_id}] {T_LOST}초 경과 → LOST VLM 2차 비동기 호출")
                    safe_img = apply_privacy_filter(frame, person_boxes, (x1, y1, x2, y2))
                    if safe_img is not None:
                        fname = f"trigger_event_{master_id}_lost.jpg"
                        cv2.imwrite(fname, safe_img)
                        bag.last_vlm_call = current_time

                        question = (
                            "이 가방은 30분 이상 같은 자리에 방치되어 있습니다. "
                            "현재 이미지에서 가방 주인으로 보이는 사람이 있나요? "
                            "없다면 DANGER, 있다면 WARNING으로 판정하세요."
                        )
                        bag._vlm_future = vlm_executor.submit(
                            call_vlm, fname, elapsed / 60, LOCATION, question,
                            "WARNING 또는 DANGER"
                        )
                        bag._vlm_context = {"stage": "LOST", "fname": fname}

            # ── 오버레이 그리기 ───────────────────────────
            renderer.draw_safe_radius(frame, cx, cy)
            renderer.draw_bag(frame, x1, y1, x2, y2, bag)

        # ── 사람 바운딩 박스 그리기 ───────────────────────
        for px1, py1, px2, py2 in person_boxes:
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            is_near = any(
                dist((pcx, pcy), b.center) < SAFE_DISTANCE
                for b in bags.values()
            )
            renderer.draw_person(frame, px1, py1, px2, py2, is_near)

        # ── 디버그 오버레이: 좌석 영역 + 좌석 표식 박스 ─────
        if DEBUG_DRAW_DETECTIONS:
            # 좌석 영역을 반투명 사각형으로 표시
            seat_overlay = frame.copy()
            for seat in SEATS:
                if seat.region == (0, 0, 0, 0):
                    continue
                sx1, sy1, sx2, sy2 = seat.region
                cv2.rectangle(seat_overlay, (sx1, sy1), (sx2, sy2), (255, 0, 255), -1)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 255), 2)
                cv2.putText(frame, f"seat_{seat.seat_id}", (sx1 + 4, sy1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(seat_overlay, 0.15, frame, 0.85, 0, frame)

            # 좌석 표식 박스 (회색 + 클래스명)
            for ix1, iy1, ix2, iy2, cls in seat_indicators:
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (180, 180, 180), 1)
                label = SEAT_INDICATOR_NAMES.get(cls, str(cls))
                cv2.putText(frame, label, (ix1, iy1 - 3),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (180, 180, 180), 1, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────
        renderer.draw_hud(frame, len(bags), len(person_boxes))

        # ── 게시판 + 대시보드 업데이트 ─────────────────────
        for b in bags.values():
            if STATE_PRIORITY.get(b.state, 0) >= STATE_PRIORITY[ST_WARNING]:
                web_status_map = {ST_WARNING: "WARNING", ST_SUSPICIOUS: "WARNING", ST_LOST: "DANGER"}
                web_status = web_status_map.get(b.state, "WARNING")
                reason = getattr(b, '_last_reason', f"상태: {b.state}")

                # 바운딩 박스 클램프
                f_h, f_w = frame.shape[:2]
                bx1 = max(0, b._bbox[0])
                by1 = max(0, b._bbox[1])
                bx2 = min(f_w, b._bbox[2])
                by2 = min(f_h, b._bbox[3])

                if bx2 > bx1 + 10 and by2 > by1 + 10:
                    snoozed = save_board_post(
                        master_id=b.master_id,
                        status=web_status,
                        reason=reason,
                        score=b.score,
                        location=LOCATION,
                        image_frame=clean_frame,
                        person_boxes=person_boxes,
                        bag_box=(bx1, by1, bx2, by2),
                    )
                    if snoozed:
                        b.reset_to_tracking(current_time)

        # ── 좌석 점유 디버그 (5초마다) ────────────────────
        if current_time - last_seat_debug >= 5.0:
            elapsed_sec = int(current_time - pipeline_start)
            global_counts = {name: 0 for name in SEAT_INDICATOR_NAMES.values()}
            for _, _, _, _, cls in seat_indicators:
                global_counts[SEAT_INDICATOR_NAMES[cls]] += 1
            bag_count = len(bags)
            counts_str = " ".join(f"{k}={v}" for k, v in global_counts.items())
            print(f"[t={elapsed_sec}s] global: bag={bag_count} {counts_str}")
            for seat in SEATS:
                if seat.region == (0, 0, 0, 0):
                    print(f"  seat_{seat.seat_id}: (영역 미설정)")
                    continue
                bag_in = sum(1 for b in bags.values() if box_in_seat(b._bbox, seat))
                ind_in = {name: 0 for name in SEAT_INDICATOR_NAMES.values()}
                for x1, y1, x2, y2, cls in seat_indicators:
                    if box_in_seat((x1, y1, x2, y2), seat):
                        ind_in[SEAT_INDICATOR_NAMES[cls]] += 1
                ind_str = " ".join(f"{k}={v}" for k, v in ind_in.items())
                print(f"  seat_{seat.seat_id}: bag={bag_in} {ind_str}")
            last_seat_debug = current_time

        # ── GC: 오래된 객체 제거 ──────────────────────────
        if current_time - last_gc > GC_INTERVAL:
            before = len(bags)
            bags = {
                tid: b for tid, b in bags.items()
                if current_time - b.last_seen < GC_MAX_AGE
            }
            removed = before - len(bags)
            if removed:
                print(f"🗑️  GC: {removed}개 오래된 객체 제거")
            last_gc = current_time

        # ── 화면 출력 (고해상도 영상 리사이즈) ────────────
        display = frame
        if DISPLAY_WIDTH and frame.shape[1] > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / frame.shape[1]
            display = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow("Intelligent Surveillance AI", display)
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