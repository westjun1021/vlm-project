"""로그/알림 파일 출력 — JSONL 로그와 관제용 latest_alert 갱신.

순수 I/O 함수만 포함 (표준 라이브러리 외 의존 없음).
"""
import os
import json
import shutil
import time


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
