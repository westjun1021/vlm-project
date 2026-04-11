import cv2
import time
import math
import os
import base64
import json
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, Future
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
#  설정 (Config)
# ============================================================
load_dotenv()

# API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API 키를 찾을 수 없습니다.")
client = OpenAI(api_key=api_key)

# 모델
YOLO_MODEL     = "yolo11n.pt"
TARGET_CLASSES  = [24, 26, 28]          # 가방류 (backpack, handbag, suitcase)
PERSON_CLASS    = 0
VLM_BACKEND     = "gpt"                # "gpt" | "local"

# 거리 / 움직임 (px)
INHERIT_TOLERANCE   = 120               # ID 상속 허용 거리 (DeepSORT ID 변경 시)
MOVEMENT_TOLERANCE  = 150               # 가방 이동 판단 거리 (스무딩 + 손떨림 + 감지 흔들림 고려)
SAFE_DISTANCE       = 150               # 핸드폰 고해상도 영상 기준 (IP카메라 전환 시 300으로 조정)
INHERIT_TIME_WINDOW = 60.0              # 초

# 상태 머신 시간 임계값 (테스트용 — 운영 시 주석 참고)
T_TRACKING  = 10    # → SUSPICIOUS  (운영: 600  = 10분)
T_WARNING   = 15    # → WARNING     (운영: 900  = 15분)
T_LOST      = 20    # → LOST        (운영: 1800 = 30분)

# 주인 판별
DWELL_OWNER_SEC  = 3    # 이상 머물면 주인 후보 (운영: 10)
DWELL_WINDOW_SEC = 15   # 최근 N초 내 기록만    (운영: 60)
PASSERBY_MAX_SEC = 3    # 이하면 행인           (운영: 5)
DWELL_RETURN_SEC = 10   # WARNING 해제에 필요한 연속 체류 시간
PERSON_GRACE_SEC = 3    # 사람 감지 유예: 이 시간 내 감지 이력 있으면 "근처에 있음"

# API 쿨다운
API_COOLDOWN = 30       # 초

# 게시판
BOARD_MIN_SCORE = 75    # 이 점수 이상일 때만 게시판에 등록

# 감지 필터링
YOLO_CONF       = 0.45     # 신뢰도 임계값 (0.35에서도 오탐 → 0.45)
MIN_BOX_AREA    = 1500      # px²: 이보다 작은 박스는 무시 (노이즈 제거)
MAX_ASPECT_RATIO = 5.0      # 가로세로 비율 제한 (길쭉한 오탐 제거)
PERSON_OVERLAP_THRESH = 0.5 # 사람 박스와 50% 이상 겹치면 오탐으로 제거

# GC
GC_INTERVAL  = 30       # 초마다 오래된 객체 정리
GC_MAX_AGE   = 120      # 초 이상 미감지 시 제거

# 디스플레이
DISPLAY_WIDTH = 960     # px: 화면 표시용 리사이즈 (None이면 원본 크기)
FRAME_SKIP    = 2       # N프레임마다 1번 처리 (1=전부, 2=절반, 3=1/3)

# 상태 상수
ST_TRACKING   = "TRACKING"
ST_SUSPICIOUS = "SUSPICIOUS"
ST_WARNING    = "WARNING"
ST_LOST       = "LOST"

# 상태별 색상 (BGR)
STATE_COLORS = {
    ST_TRACKING:   (0, 200, 0),     # 초록
    ST_SUSPICIOUS: (0, 200, 255),   # 주황
    ST_WARNING:    (0, 140, 255),   # 진한 주황
    ST_LOST:       (0, 0, 255),     # 빨강
}

# 대시보드 우선순위 (높을수록 우선 표시)
STATE_PRIORITY = {
    ST_TRACKING:   0,
    ST_SUSPICIOUS: 1,
    ST_WARNING:    2,
    ST_LOST:       3,
}


# ============================================================
#  유틸리티
# ============================================================
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def box_overlap_ratio(box_a, box_b):
    """box_a가 box_b에 얼마나 포함되는지 비율 (0~1). box = (x1,y1,x2,y2)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    return inter / area_a


def apply_privacy_filter(frame, person_boxes, target_box):
    """사람 영역 블러 + 가방 주변 크롭"""
    h, w = frame.shape[:2]
    out = frame.copy()
    for px1, py1, px2, py2 in person_boxes:
        roi = out[py1:py2, px1:px2]
        if roi.size:
            out[py1:py2, px1:px2] = cv2.GaussianBlur(roi, (51, 51), 0)
    tx1, ty1, tx2, ty2 = target_box
    pad = 150
    cx1, cy1 = max(0, tx1 - pad), max(0, ty1 - pad)
    cx2, cy2 = min(w, tx2 + pad), min(h, ty2 + pad)
    crop = out[cy1:cy2, cx1:cx2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        print("⚠️  크롭 이미지 너무 작음 → API 호출 건너뜀")
        return None
    return crop


def save_log(image_path, elapsed, location, state, result_json, backend):
    os.makedirs("logs", exist_ok=True)
    entry = {
        "image":    image_path,
        "elapsed":  round(elapsed, 1),
        "location": location,
        "state":    state,
        "status":   result_json.get("status", ""),
        "reason":   result_json.get("reason", ""),
        "backend":  backend,
        "time":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("logs/detection_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_alert(master_id, state, result_json, score, location, filename):
    """기존 관제용 (호환성 유지)"""
    os.makedirs("static", exist_ok=True)
    shutil.copy(filename, "static/latest_alert.jpg")
    data = {
        "id":       master_id,
        "state":    state,
        "status":   result_json.get("status", state),
        "reason":   result_json.get("reason", ""),
        "score":    round(score, 1),
        "location": location,
        "time":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("static/latest_alert.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# 게시판 게시 관리
_posted_ids: dict[int, str] = {}   # master_id → post_id 매핑


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
        except Exception:
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
            except Exception:
                pass
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


# ============================================================
#  VLM 백엔드
# ============================================================
def _call_gpt(image_path, elapsed_min, location, question, valid_statuses):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    system_prompt = (
        "당신은 캠퍼스 분실물 관제 AI입니다. "
        "얼굴/체형은 블러 처리되었습니다. "
        "반드시 JSON 형식으로만 응답하세요. "
        f'형식: {{"status": "{valid_statuses}", "reason": "판단 근거 (한국어)"}}'
    )
    user_prompt = (
        f"[상황] 장소: {location} / 가방 방치 경과 시간: 약 {elapsed_min:.0f}분\n"
        f"[질문] {question}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError("VLM 응답 비어있음")
    return raw


def _call_local(image_path, elapsed_min, location, question, valid_statuses):
    """2학기: 파인튜닝 로컬 VLM 서버"""
    import requests
    url = os.getenv("LOCAL_VLM_URL", "http://localhost:8000/predict")
    with open(image_path, "rb") as f:
        r = requests.post(
            url,
            files={"image": f},
            data={"elapsed_min": elapsed_min, "location": location,
                  "question": question, "valid_statuses": valid_statuses},
            timeout=30,
        )
    r.raise_for_status()
    return r.text


def call_vlm(image_path, elapsed_min, location, question, valid_statuses):
    if VLM_BACKEND == "gpt":
        return _call_gpt(image_path, elapsed_min, location, question, valid_statuses)
    return _call_local(image_path, elapsed_min, location, question, valid_statuses)


# ============================================================
#  TrackedBag — 개별 가방 객체의 상태 머신
# ============================================================
class TrackedBag:
    """하나의 추적 대상(가방)에 대한 상태·타이머·체류 기록을 캡슐화"""

    def __init__(self, master_id, center, size, start_time):
        self.master_id       = master_id
        self.start_time      = start_time
        self.last_seen       = start_time
        self.center          = center
        self.anchor_center   = center
        self.size            = size
        self.state           = ST_TRACKING

        # 체류 기록
        self.dwell_log: list[tuple[float, float]] = []
        self.person_near_since: float | None       = None
        self.person_last_detected: float           = 0.0   # 마지막으로 사람이 감지된 시각

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

    # ── 체류 기록 갱신 ────────────────────────────────────
    def update_dwell(self, person_boxes, bag_center, bag_box, current_time):
        """사람-가방 근접 여부 갱신. 중심 거리 OR 박스 겹침으로 판단."""
        person_is_near = False

        for px1, py1, px2, py2 in person_boxes:
            # 방법 1: 중심점 거리
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            if dist(bag_center, (pcx, pcy)) < SAFE_DISTANCE:
                person_is_near = True
                break
            # 방법 2: 바운딩 박스 겹침 (사람 몸이 가방과 닿아 있으면)
            if box_overlap_ratio(bag_box, (px1, py1, px2, py2)) > 0 or \
               box_overlap_ratio((px1, py1, px2, py2), bag_box) > 0:
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


# ============================================================
#  오버레이 렌더러 — 영상 위 시각화
# ============================================================
class OverlayRenderer:
    """프레임 위에 추적 정보를 그리는 전담 클래스"""

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

    @staticmethod
    def draw_bag(frame, x1, y1, x2, y2, bag: TrackedBag):
        """가방 바운딩 박스 + 상태 라벨 + 경과 시간"""
        color = STATE_COLORS.get(bag.state, (200, 200, 200))
        thickness = 2 if bag.state == ST_TRACKING else 3

        # 바운딩 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 상태 라벨 배경
        elapsed_sec = int(bag.elapsed)
        label = f"ID:{bag.master_id} {bag.state} {elapsed_sec}s"
        (tw, th), _ = cv2.getTextSize(label, OverlayRenderer.FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5),
                    OverlayRenderer.FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 점수 바 (박스 아래)
        bar_w = x2 - x1
        bar_h = 4
        filled = int(bar_w * bag.score / 100)
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


# ============================================================
#  메인 파이프라인
# ============================================================
def main():
    # ── 초기화 ────────────────────────────────────────────
    yolo    = YOLO(YOLO_MODEL)
    tracker = DeepSort(max_age=150, n_init=3, max_iou_distance=0.9)
    cap     = cv2.VideoCapture("test_video4.mp4")

    LOCATION = "캠퍼스 열람실"   # 고정값 (운영 시에는 카메라별로 다르게 설정)

    bags: dict[int, TrackedBag] = {}   # track_id → TrackedBag
    last_gc       = 0.0
    frame_count   = 0
    renderer      = OverlayRenderer()
    vlm_executor  = ThreadPoolExecutor(max_workers=2)  # VLM 비동기 호출용

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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    vlm_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()