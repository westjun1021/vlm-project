"""VLM 백엔드 — GPT-4o (OpenAI) 또는 로컬 파인튜닝 서버 호출.

OpenAI 클라이언트는 모듈 내부에서만 사용 (외부에 노출 안 함).
호출자는 `call_vlm`만 import하면 충분하다.
"""
import os
import base64
from openai import OpenAI

from config import VLM_BACKEND, api_key

# 모듈 내부 전용 클라이언트
_client = OpenAI(api_key=api_key)


def _call_gpt(image_path, elapsed_min, location, question, valid_statuses, seat_info=""):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    system_prompt = (
        "당신은 캠퍼스 분실물 관제 AI입니다. "
        "얼굴/체형은 블러 처리되었습니다. "
        "반드시 JSON 형식으로만 응답하세요. "
        f'형식: {{"status": "{valid_statuses}", "reason": "판단 근거 (한국어)"}}'
    )
    seat_line = f"\n[좌석 점유] {seat_info}" if seat_info else ""
    user_prompt = (
        f"[상황] 장소: {location} / 물건 방치 경과 시간: 약 {elapsed_min:.0f}분"
        f"{seat_line}\n"
        f"[질문] {question}"
    )
    resp = _client.chat.completions.create(
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


def _call_local(image_path, elapsed_min, location, question, valid_statuses, seat_info=""):
    """2학기: 파인튜닝 로컬 VLM 서버"""
    import requests
    url = os.getenv("LOCAL_VLM_URL", "http://localhost:8000/predict")
    with open(image_path, "rb") as f:
        r = requests.post(
            url,
            files={"image": f},
            data={"elapsed_min": elapsed_min, "location": location,
                  "question": question, "valid_statuses": valid_statuses,
                  "seat_info": seat_info},
            timeout=30,
        )
    r.raise_for_status()
    return r.text


def call_vlm(image_path, elapsed_min, location, question, valid_statuses, seat_info=""):
    if VLM_BACKEND == "gpt":
        return _call_gpt(image_path, elapsed_min, location, question, valid_statuses, seat_info)
    return _call_local(image_path, elapsed_min, location, question, valid_statuses, seat_info)
