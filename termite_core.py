"""
termite_core.py
----------------------------------------------------------
GUI/카메라와 무관한 순수 이미지처리·추적 로직.
(테스트가 쉽도록 GUI 코드와 분리했습니다)

포함 기능:
- detect_petri_dish   : 패트리디쉬(원형 용기) 자동 검출
- make_circular_mask  : 원형 ROI 마스크 생성
- detect_objects      : 기준영상 대비 차분으로 개체(흰개미) 검출
- CentroidTracker     : 중심점 기반 개체 ID 추적기
- classify_state      : 활동 / 관찰중 / 사멸의심 상태 분류
----------------------------------------------------------
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 상태별 표시 색상 (BGR, OpenCV 기준)
STATE_ACTIVE = "활동"
STATE_OBSERVING = "관찰중"
STATE_DEAD_SUSPECT = "사멸의심"

STATE_COLORS = {
    STATE_ACTIVE: (0, 200, 0),       # 초록
    STATE_OBSERVING: (0, 200, 255),  # 노랑/주황
    STATE_DEAD_SUSPECT: (0, 0, 220), # 빨강
}


# ----------------------------------------------------------------
# 1. 패트리디쉬(원형 용기) 검출
# ----------------------------------------------------------------
def detect_petri_dish(gray_img: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """그레이스케일 이미지에서 가장 그럴듯한 원(패트리디쉬)을 찾는다.

    Returns:
        (cx, cy, r) 또는 검출 실패 시 None
    """
    h, w = gray_img.shape[:2]
    blurred = cv2.medianBlur(gray_img, 5)
    min_r = max(10, int(min(h, w) * 0.20))
    max_r = int(min(h, w) * 0.49)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(h, w),
        param1=60, param2=40, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    cx, cy, r = max(circles, key=lambda c: c[2])  # 가장 큰 원 선택
    return int(cx), int(cy), int(r)


def make_circular_mask(shape: Tuple[int, ...], cx: int, cy: int, r: int) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r), 255, thickness=-1)
    return mask


# ----------------------------------------------------------------
# 2. 기준영상 대비 개체 검출 (배경 차분)
# ----------------------------------------------------------------
def detect_objects(
    gray_current: np.ndarray,
    gray_reference: np.ndarray,
    roi_mask: Optional[np.ndarray],
    min_area: int = 15,
    max_area: int = 2000,
    sensitivity: int = 25,
) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """기준(빈 여과지) 영상과의 차이를 이용해 개체 후보를 검출한다.

    Args:
        sensitivity: 차분 이진화 임계값 (낮을수록 민감 = 작은 차이도 검출)

    Returns:
        detections: [(cx, cy, area), ...]
        thresh: 디버깅/시각화용 이진 마스크
    """
    if gray_current.shape != gray_reference.shape:
        gray_current = cv2.resize(gray_current, (gray_reference.shape[1], gray_reference.shape[0]))

    diff = cv2.absdiff(gray_current, gray_reference)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(diff, int(sensitivity), 255, cv2.THRESH_BINARY)

    if roi_mask is not None:
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        detections.append((cx, cy, area))

    return detections, thresh


# ----------------------------------------------------------------
# 3. 중심점 기반 개체 추적기
# ----------------------------------------------------------------
@dataclass
class TrackedObject:
    object_id: int
    centroid: Tuple[float, float]
    first_seen: float
    last_seen: float
    last_move_time: float
    missed_frames: int = 0
    path: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)

    def __post_init__(self):
        if not self.path:
            self.path.append((self.first_seen, self.centroid[0], self.centroid[1]))


def _pairwise_dist(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    diff = a_arr[:, None, :] - b_arr[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


class CentroidTracker:
    """단순 최근접 중심점 매칭 기반 다중개체 추적기.

    참고: Hungarian 알고리즘이 아닌 그리디(탐욕) 매칭을 사용하므로,
    개체가 서로 심하게 겹치는 순간에는 ID가 바뀔 수 있습니다.
    (사용설명서의 '겹침/가림 주의' 안내와 일치)
    """

    def __init__(self, max_distance: float = 60.0, max_missed_frames: int = 20,
                 move_threshold_px: float = 3.0):
        self.next_id = 0
        self.objects: Dict[int, TrackedObject] = {}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.move_threshold_px = move_threshold_px

    def _register(self, centroid: Tuple[float, float], timestamp: float) -> int:
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = TrackedObject(
            object_id=oid, centroid=centroid,
            first_seen=timestamp, last_seen=timestamp, last_move_time=timestamp,
        )
        return oid

    def _deregister_stale(self):
        stale = [oid for oid, o in self.objects.items() if o.missed_frames > self.max_missed_frames]
        for oid in stale:
            del self.objects[oid]

    def update(self, detections: List[Tuple[float, float]], timestamp: float) -> Dict[int, TrackedObject]:
        if not self.objects:
            for c in detections:
                self._register(c, timestamp)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]

        if len(detections) == 0:
            for oid in object_ids:
                self.objects[oid].missed_frames += 1
            self._deregister_stale()
            return self.objects

        d = _pairwise_dist(object_centroids, detections)
        used_rows, used_cols = set(), set()

        # 각 트랙(row)을 가장 가까운 검출(col)에 그리디하게 매칭
        row_order = d.min(axis=1).argsort()
        for row in row_order:
            col = int(d[row].argmin())
            if row in used_rows or col in used_cols:
                continue
            if d[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            obj = self.objects[oid]
            old_centroid = obj.centroid
            new_centroid = detections[col]
            moved = float(np.hypot(new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1]))

            obj.centroid = new_centroid
            obj.last_seen = timestamp
            obj.missed_frames = 0
            obj.path.append((timestamp, new_centroid[0], new_centroid[1]))
            if moved >= self.move_threshold_px:
                obj.last_move_time = timestamp

            used_rows.add(row)
            used_cols.add(col)

        for i, oid in enumerate(object_ids):
            if i not in used_rows:
                self.objects[oid].missed_frames += 1

        for j, c in enumerate(detections):
            if j not in used_cols:
                self._register(c, timestamp)

        self._deregister_stale()
        return self.objects


# ----------------------------------------------------------------
# 4. 상태 분류
# ----------------------------------------------------------------
def classify_state(obj: TrackedObject, now: float, no_movement_threshold_sec: float,
                    observing_grace_sec: float = 5.0) -> str:
    """개체의 현재 상태를 분류한다.

    - 갓 등장한 개체(관찰 유예시간 이내)            -> 관찰중
    - 최근(유예시간 이내)에 실제로 움직인 개체        -> 활동
    - 움직이지 않은 지 무이동 판정시간을 넘지 않은 개체 -> 관찰중
    - 무이동 판정시간을 넘긴 개체                    -> 사멸의심 (사망 확정 아님)
    """
    age = now - obj.first_seen
    since_move = now - obj.last_move_time

    if age < observing_grace_sec:
        return STATE_OBSERVING
    if since_move < observing_grace_sec:
        return STATE_ACTIVE
    if since_move < no_movement_threshold_sec:
        return STATE_OBSERVING
    return STATE_DEAD_SUSPECT
