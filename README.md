# VLM Project — 실시간 유실물 감지 시스템

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLO-v11-00FFFF?logo=ultralytics&logoColor=black" />
  <img src="https://img.shields.io/badge/DeepSORT-Tracker-orange" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o_Vision-412991?logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Web_Server-009688?logo=fastapi&logoColor=white" />
</p>

---

## 📌 프로젝트 개요

CCTV 영상에서 **가방·캐리어 등 유실물 의심 상황을 자동 감지**하는 AI 파이프라인입니다.  
YOLOv11 객체 탐지 + DeepSORT 다중 추적 + GPT-4o Vision 상황 판단을 결합하여,  
소지자 이탈 여부·소유자 판별·장시간 방치 여부를 상태 머신 기반으로 분류합니다.

---

## 🎯 주요 기능

| 기능 | 설명 |
|------|------|
| 객체 탐지 | YOLOv11 — backpack / handbag / suitcase 실시간 감지 |
| 다중 추적 | DeepSORT — 여러 가방을 프레임 간 ID 유지하며 추적 |
| 상태 머신 | TRACKING → SUSPICIOUS → WARNING → LOST 단계별 판별 |
| 소유자 판별 | 주변 체류 시간 기반으로 소유자 vs 행인 자동 구분 |
| VLM 검증 | 의심 상황 발생 시 GPT-4o Vision에 이미지 전송 → 재확인 |
| 웹 대시보드 | FastAPI + HTML 템플릿 기반 실시간 모니터링 화면 |

---

## 🧠 시스템 구조

```
Camera Input (영상)
    ↓
YOLOv11  ─── 가방류 + 사람 탐지
    ↓
DeepSORT ─── ID 유지 / 궤적 추적
    ↓
State Machine
    ├── TRACKING   : 소지 정상 추적
    ├── SUSPICIOUS : 소지자 이탈 감지
    ├── WARNING    : 장시간 방치 경고
    └── LOST       : 유실물 확정
    ↓
GPT-4o Vision API (의심 단계에서 크롭 이미지 전송)
    ↓
Web Dashboard (FastAPI / templates/index.html)
```

---

## 📁 파일 구조

```
vlm-project/
├── main_pipeline.py   # 핵심 파이프라인 (탐지·추적·상태 머신·VLM 호출)
├── web_server.py      # FastAPI 웹 서버 (대시보드 제공)
├── templates/
│   └── index.html    # 실시간 모니터링 대시보드 UI
├── .env               # OPENAI_API_KEY (로컬 전용, 커밋 제외)
├── .gitignore
└── README.md
```

---

## ⚙️ 핵심 파라미터

```python
# 상태 전환 시간 (운영 기준)
T_TRACKING  = 600   # 10분 → SUSPICIOUS
T_WARNING   = 900   # 15분 → WARNING
T_LOST      = 1800  # 30분 → LOST

# 탐지 필터
YOLO_CONF        = 0.45   # 신뢰도 임계값
MIN_BOX_AREA     = 1500   # 최소 박스 면적(px²)
MAX_ASPECT_RATIO = 5.0    # 길쭉한 오탐 제거

# VLM 호출
API_COOLDOWN     = 30     # 초 (과호출 방지)
BOARD_MIN_SCORE  = 75     # 게시판 등록 최소 신뢰도 점수
```

---

## 🚀 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt   # ultralytics, deep-sort-realtime, openai, fastapi, python-dotenv

# 2. 환경변수 설정
echo "OPENAI_API_KEY=your_key_here" > .env

# 3. 파이프라인 단독 실행
python main_pipeline.py

# 4. 웹 대시보드 실행
python web_server.py
# → http://localhost:8000 접속
```

---

## 🛠 기술 스택

- **AI/ML**: YOLOv11 (Ultralytics), DeepSORT, OpenAI GPT-4o Vision
- **백엔드**: FastAPI, Python 3.9+
- **라이브러리**: OpenCV, NumPy, python-dotenv, concurrent.futures

---

## 📌 향후 개선 사항

- [ ] LiDAR / 깊이 카메라 연동으로 3D 위치 추적
- [ ] 다중 카메라 뷰 통합 (Cross-Camera Re-ID)
- [ ] 알림 시스템 연동 (SMS, Slack, MQTT)
- [ ] ONNX 변환으로 엣지 디바이스 배포

---

## 👤 개발자

**westjun1021** — Computer Vision & Backend Developer  
Email: tjwns4603@gmail.com
