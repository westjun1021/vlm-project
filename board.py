"""분실물 게시판 — JSON/이미지 파일 기반 게시글 관리.

_posted_ids는 모듈 내부 전역으로 캡슐화 (외부에서 직접 접근하지 않음).
호출자는 save_board_post / remove_board_post / _restore_posted_ids만 사용.
"""
import os
import json
import glob
import time
import traceback
import cv2

from config import BOARD_MIN_SCORE
from utils import apply_privacy_filter


# 게시판 게시 관리 — master_id → post_id 매핑 (모듈 내부 전용)
_posted_ids: dict[int, str] = {}


def _restore_posted_ids():
    """파이프라인 재시작 시 기존 board JSON에서 _posted_ids 복원.
    없으면 중복 등록 / 주인 복귀 시 삭제 누락이 발생한다."""
    board_dir = "static/board"
    if not os.path.isdir(board_dir):
        return
    restored = 0
    for path in glob.glob(os.path.join(board_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            master_id = data.get("id")
            post_id   = data.get("post_id")
            if master_id is not None and post_id:
                _posted_ids[int(master_id)] = post_id
                restored += 1
        except Exception as e:
            print(f"⚠️ [board:_restore_posted_ids] {os.path.basename(path)} 로드 실패 — {type(e).__name__}: {e}")
            continue
    if restored:
        print(f"📂 [복원] 기존 게시글 {restored}건 _posted_ids 로드 완료")


def save_board_post(master_id, status, reason, score, location, image_frame,
                    person_boxes, bag_box):
    """분실물 게시판에 등록 또는 업데이트
    - 신규: score >= BOARD_MIN_SCORE 또는 DANGER일 때만 등록
    - 기존: score/status 변동 시 업데이트
    """
    board_dir = "static/board"
    os.makedirs(board_dir, exist_ok=True)

    # 이미 등록된 글이면 → 업데이트
    if master_id in _posted_ids:
        post_id = _posted_ids[master_id]
        json_path = os.path.join(board_dir, f"{post_id}.json")
        if not os.path.exists(json_path):
            # 웹에서 '찾아감' 처리로 파일이 삭제된 경우 → 추적 스누즈 신호 반환
            print(f"✅ [ID: {master_id}] 웹에서 '찾아감' 처리됨 → 추적 타이머 리셋")
            del _posted_ids[master_id]
            return True  # 메인 루프에 스누즈 신호
        else:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                # score나 status가 변했으면 갱신
                if abs(existing.get("score", 0) - score) >= 5 or existing.get("status") != status:
                    existing["score"]  = round(score, 1)
                    existing["status"] = status
                    existing["reason"] = reason
                    existing["time"]   = time.strftime("%Y-%m-%d %H:%M:%S")
                    tmp_path = json_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(existing, f, ensure_ascii=False)
                    os.replace(tmp_path, json_path)
                    # 이미지도 갱신
                    safe_img = apply_privacy_filter(image_frame, person_boxes, bag_box)
                    if safe_img is not None:
                        cv2.imwrite(os.path.join(board_dir, f"{post_id}.jpg"), safe_img)
            except Exception as e:
                print(f"⚠️ [board:save_board_post] master_id={master_id} 업데이트 실패 — {type(e).__name__}: {e}")
                traceback.print_exc()
            return

    # 신규 등록 — 점수 기준 미달이면 스킵
    if score < BOARD_MIN_SCORE and status != "DANGER":
        return

    post_id = f"lost_{master_id}_{int(time.time())}"
    _posted_ids[master_id] = post_id

    # 크롭 이미지 저장
    safe_img = apply_privacy_filter(image_frame, person_boxes, bag_box)
    if safe_img is not None:
        cv2.imwrite(os.path.join(board_dir, f"{post_id}.jpg"), safe_img)

    # 게시글 JSON 저장 (Atomic Write — 웹 서버와의 동시성 충돌 방지)
    post_data = {
        "post_id":  post_id,
        "id":       master_id,
        "status":   status,
        "reason":   reason,
        "score":    round(score, 1),
        "location": location,
        "time":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = os.path.join(board_dir, f"{post_id}.json")
    tmp_path  = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(post_data, f, ensure_ascii=False)
    os.replace(tmp_path, json_path)

    print(f"📌 [게시판] 분실물 등록: {post_id} ({status}, {score:.0f}점)")


def remove_board_post(master_id):
    """주인 복귀 시 게시글 삭제"""
    board_dir = "static/board"
    if master_id in _posted_ids:
        post_id = _posted_ids.pop(master_id)
        for ext in (".json", ".jpg"):
            path = os.path.join(board_dir, f"{post_id}{ext}")
            if os.path.exists(path):
                os.remove(path)
        print(f"🗑️  [게시판] 분실물 삭제 (주인 복귀): {post_id}")
