# event_clip.py — 이벤트 클립 저장 (별도 스레드 mp4 인코딩)
import os
import cv2
import traceback
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from config import EVENTS_DIR, EVENT_POST_SECONDS, FRAME_SKIP

# ============================================================
#  이벤트 클립 저장 (페이즈 7 단계 3)
#  - frame_buffer에서 과거 N초 추출 + 이벤트 후 M초 실시간 녹화
#  - mp4 인코딩은 별도 스레드 (메인 루프 차단 안 함)
# ============================================================


class EventClipRecorder:
    """
    활성 녹화 세션을 관리. main_pipeline이 매 프레임마다 update()를 호출하면
    녹화 중인 모든 세션이 추가 프레임을 누적함. 시간 다 되면 자동 종료 + 저장.
    """

    def __init__(self, executor: ThreadPoolExecutor):
        self._executor = executor
        # 활성 녹화 세션: list of dicts
        # {start_time, end_time, master_id, state_label, prefix_frames, post_frames}
        self._active_sessions: list = []

    def trigger(self, master_id: int, state_label: str,
                frame_buffer: deque, current_time: float):
        """
        이벤트 트리거. frame_buffer 스냅샷 (과거 10초) + 새 세션 시작 (이후 10초).

        Args:
            master_id: TrackedItem.master_id
            state_label: "WARNING" | "LOST" | "RECOVERED"
            frame_buffer: main_pipeline의 frame_buffer deque [(timestamp, frame), ...]
            current_time: time.time()
        """
        # 과거 프레임 스냅샷 (얕은 복사 — list로 변환)
        # frame은 numpy array 참조만 들고감 (메모리 효율).
        # 어차피 frame_buffer는 maxlen으로 자동 폐기되므로 새 list로 분리.
        prefix_frames = [(t, f) for (t, f) in frame_buffer]

        session = {
            "start_time": current_time,
            "end_time": current_time + EVENT_POST_SECONDS,
            "master_id": master_id,
            "state_label": state_label,
            "prefix_frames": prefix_frames,
            "post_frames": [],   # update()로 누적
        }
        self._active_sessions.append(session)
        print(f"🎬 [클립] {state_label} master_id={master_id} 녹화 시작 "
              f"(전 {len(prefix_frames)} 프레임 + 후 {EVENT_POST_SECONDS}초)")

    def update(self, current_time: float, clean_frame):
        """
        매 프레임 호출. 활성 세션에 frame 추가. 종료 시 자동 저장.
        clean_frame: 오버레이 전 프레임 (frame_buffer와 동일 소스)
        """
        if not self._active_sessions:
            return

        still_active = []
        for session in self._active_sessions:
            if current_time < session["end_time"]:
                # 아직 녹화 중
                session["post_frames"].append((current_time, clean_frame))
                still_active.append(session)
            else:
                # 녹화 끝 — 저장 위임
                self._executor.submit(_save_clip, session)
        self._active_sessions = still_active

    def flush(self):
        """
        프로그램 종료 시 남은 세션 즉시 저장 (잘릴 수도).
        """
        for session in self._active_sessions:
            self._executor.submit(_save_clip, session)
        self._active_sessions = []


def _save_clip(session: dict):
    """
    별도 스레드에서 실행. prefix_frames + post_frames 합쳐서 mp4 작성.
    """
    try:
        master_id = session["master_id"]
        state_label = session["state_label"]
        start_time = session["start_time"]

        # 폴더 생성 (날짜별)
        date_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d")
        time_str = datetime.fromtimestamp(start_time).strftime("%H%M%S")
        out_dir = os.path.join(EVENTS_DIR, date_str)
        os.makedirs(out_dir, exist_ok=True)

        # 파일명
        filename = f"{time_str}_{state_label}_{master_id}.mp4"
        out_path = os.path.join(out_dir, filename)

        # 전체 프레임 합치기
        all_frames = session["prefix_frames"] + session["post_frames"]
        if not all_frames:
            print(f"⚠️  [클립] {filename} 프레임 0개 — 저장 안 함")
            return

        # 첫 프레임으로 크기/fps 결정
        _, first_frame = all_frames[0]
        h, w = first_frame.shape[:2]
        fps = 30 / FRAME_SKIP   # 실효 FPS

        # VideoWriter 생성
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"❌ [클립] VideoWriter 열기 실패: {out_path}")
            return

        for _, frame in all_frames:
            writer.write(frame)
        writer.release()

        # 결과
        duration = len(all_frames) / fps
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"💾 [클립 저장] {filename} ({len(all_frames)} 프레임, "
              f"{duration:.1f}초, {size_mb:.1f}MB)")

    except Exception as e:
        print(f"❌ [클립 저장] master_id={session.get('master_id')} 실패 — "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
