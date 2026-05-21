"""순수 유틸리티 함수 — 기하 연산과 이미지 전처리.

config나 다른 프로젝트 모듈에 의존하지 않는 stateless 함수만 포함.
"""
import math
import cv2


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
