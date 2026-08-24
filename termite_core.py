"""
termite_core.py
----------------------------------------------------------
GUI/카메라와 무관한 순수 이미지처리·추적 로직.
(테스트가 쉽도록 GUI 코드와 분리했습니다)

포함 기능:
- detect_petri_dish   : 패트리디쉬(원형 용기) 자동 검출
- make_circular_mask  : 원형 ROI 마스크 생성
- detect_objects      : 기준영상 대비 차분으로 개체(흰개미) 검출 (+ 붙은 개체 분리)
- CentroidTracker     : 중심점 기반 개체 ID 추적기 (전역 최적 그리디 매칭)
- classify_state      : 활동 / 관찰중 / 사멸의심 상태 분류
----------------------------------------------------------
"""

from dataclasses import dataclass, field
from math import hypot
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


def make_circular_mask(
    shape: Tuple[int, ...], cx: int, cy: int, r: int, inner_ratio: float = 1.0
) -> np.ndarray:
    """원형 ROI 마스크를 만든다.

    inner_ratio(<1.0)를 주면 패트리디쉬 테두리(그림자/반사가 생기기 쉬운 가장자리)를
    검출 대상에서 제외해 오검출을 줄일 수 있다. 기본값 1.0은 기존 동작과 동일하다.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    effective_r = max(1, int(round(r * inner_ratio)))
    cv2.circle(mask, (int(cx), int(cy)), effective_r, 255, thickness=-1)
    return mask


# ----------------------------------------------------------------
# 2. 기준영상 대비 개체 검출 (배경 차분 + 붙은 개체 분리)
# ----------------------------------------------------------------
def _split_touching_contour(
    contour: np.ndarray,
    frame_shape: Tuple[int, int],
    min_area: float,
    split_area_factor: float,
    split_peak_ratio: float,
    split_max_parts: int,
) -> List[np.ndarray]:
    """서로 붙어있는(겹친) 개체가 하나의 큰 컨투어로 검출된 경우,
    거리변환(distance transform) 피크를 씨앗(seed)으로 삼아 watershed로 분리한다.

    분리에 실패하거나(피크가 하나뿐 등) 애매하면 원래 컨투어를 그대로 반환한다.
    """
    area = cv2.contourArea(contour)
    if area < min_area * split_area_factor:
        return [contour]

    x, y, w, h = cv2.boundingRect(contour)
    pad = 3
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(frame_shape[1], x + w + pad)
    bottom = min(frame_shape[0], y + h + pad)
    local_w, local_h = right - left, bottom - top
    if local_w < 5 or local_h < 5:
        return [contour]

    local_contour = contour.copy()
    local_contour[:, 0, 0] -= left
    local_contour[:, 0, 1] -= top
    component = np.zeros((local_h, local_w), dtype=np.uint8)
    cv2.drawContours(component, [local_contour], -1, 255, thickness=-1)

    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak < 1.5:
        return [contour]

    cores = np.uint8(distance >= peak * split_peak_ratio) * 255
    cores = cv2.morphologyEx(
        cores, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
    seeds = [i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= 3]
    if len(seeds) < 2:
        return [contour]
    # 씨앗(피크)이 너무 많으면 '개체 여러 마리가 겹친 덩어리'가 아니라 헝겊 주름/종이
    # 테두리처럼 요철이 많은 이물질일 가능성이 높다. 이런 경우 잘게 쪼개면 가짜 개체가
    # 무더기로 생기므로, 분리를 포기하고 원본 컨투어를 그대로 둔다(면적/원형도 필터가
    # 뒤에서 걸러낸다).
    if len(seeds) > split_max_parts * 3:
        return [contour]
    seeds.sort(key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    seeds = seeds[:split_max_parts]

    markers = np.zeros_like(labels, dtype=np.int32)
    markers[component == 0] = 1
    for marker_id, label_id in enumerate(seeds, start=2):
        markers[labels == label_id] = marker_id

    normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    surface = cv2.cvtColor(255 - normalized, cv2.COLOR_GRAY2BGR)
    cv2.watershed(surface, markers)

    parts: List[np.ndarray] = []
    min_part_area = min_area * 0.72
    for marker_id in range(2, 2 + len(seeds)):
        part_mask = np.uint8((markers == marker_id) & (component > 0)) * 255
        part_contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not part_contours:
            continue
        part = max(part_contours, key=cv2.contourArea)
        if cv2.contourArea(part) < min_part_area:
            continue
        part[:, 0, 0] += left
        part[:, 0, 1] += top
        parts.append(part)

    if len(parts) < 2:
        return [contour]
    if sum(cv2.contourArea(p) for p in parts) < area * 0.55:
        return [contour]
    return parts


def detect_objects(
    gray_current: np.ndarray,
    gray_reference: np.ndarray,
    roi_mask: Optional[np.ndarray],
    min_area: int = 15,
    max_area: int = 2000,
    sensitivity: int = 25,
    split_touching: bool = True,
    split_area_factor: float = 2.2,
    split_peak_ratio: float = 0.62,
    split_max_parts: int = 4,
    cluster_area_multiplier: float = 3.0,
    min_circularity: float = 0.35,
) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """기준(빈 여과지) 영상과의 차이를 이용해 개체 후보를 검출한다.

    Args:
        sensitivity: 차분 이진화 임계값 (낮을수록 민감 = 작은 차이도 검출)
        split_touching: 서로 붙어있는 개체를 watershed로 분리할지 여부
        cluster_area_multiplier: watershed 분리가 실패한(피크가 하나뿐이라 나뉘지 않은)
            큰 뭉치는 원래대로라면 max_area를 넘어 완전히 버려져 '검출 자체가 안 되는'
            문제가 생긴다. 이런 미분리 뭉치는 max_area의 이 배수까지는 최소 1개체로라도
            검출 목록에 남겨서, 화면에서 사라지는 대신 하나로 뭉쳐 보이게 한다.
        min_circularity: 4π·면적/둘레² 원형도 최소값(0~1, 완전한 원=1). 흰개미는 대체로
            둥글둥글한 타원형이지만, 헝겊 주름·종이 테두리 같은 이물질은 가늘고 구불구불해
            원형도가 매우 낮다(대개 0.1 이하). 이 값 미만인 컨투어는 후보에서 제외한다.

    Returns:
        detections: [(cx, cy, area), ...]  (x좌표 기준 정렬, 재현성 확보)
        thresh: 디버깅/시각화용 이진 마스크
    """
    if gray_current.shape != gray_reference.shape:
        gray_current = cv2.resize(gray_current, (gray_reference.shape[1], gray_reference.shape[0]))

    current_background = cv2.GaussianBlur(gray_current, (0, 0), 21)
    reference_background = cv2.GaussianBlur(gray_reference, (0, 0), 21)
    normalized_current = cv2.divide(gray_current, current_background, scale=128)
    normalized_reference = cv2.divide(gray_reference, reference_background, scale=128)
    diff = cv2.absdiff(normalized_current, normalized_reference)
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
        raw_area = cv2.contourArea(c)
        raw_perimeter = cv2.arcLength(c, True)
        raw_circularity = (
            (4.0 * np.pi * raw_area / (raw_perimeter * raw_perimeter)) if raw_perimeter > 0 else 0.0
        )
        # watershed는 얇고 구불구불한 선(헝겊 주름, 종이 테두리 등)도 여러 조각으로
        # 쪼개버릴 수 있는데, 그 조각 하나하나는 짧고 둥글둥글해서 개별 원형도 검사를
        # 통과해버린다. 그래서 분리를 시도하기 '전' 원본 덩어리 자체의 원형도를 먼저
        # 확인해, 애초에 개체 뭉치라고 보기 힘든 형태(원형도가 매우 낮음)면 분리 자체를
        # 시도하지 않고 통째로 버린다. (실흰개미 여러 마리가 겹친 덩어리는 이보다는
        # 훨씬 뭉툭한 형태라 이 문턱값을 넘는다)
        if raw_area >= min_area and raw_circularity < min_circularity * 0.4:
            continue
        candidates = (
            _split_touching_contour(
                c, thresh.shape, min_area, split_area_factor, split_peak_ratio, split_max_parts
            )
            if split_touching
            else [c]
        )
        # split_touching이 시도됐지만 뚜렷한 두 번째 피크가 없어 분리에 실패한 경우
        # (candidates가 원본 컨투어 1개 그대로) -> 여러 개체가 뭉쳐서 하나의 큰 덩어리로
        # 남은 것일 수 있다. 이런 미분리 뭉치를 max_area 기준으로 완전히 버리면 화면에서
        # 개체가 통째로 사라져 '인식이 안 되는' 것처럼 보이므로, 배수 한도 내에서는
        # 최소 1개체로라도 남긴다.
        unresolved_cluster = (
            split_touching and len(candidates) == 1 and raw_area >= min_area * split_area_factor
        )
        for candidate in candidates:
            area = cv2.contourArea(candidate)
            if area < min_area:
                continue
            if area > max_area:
                if not (unresolved_cluster and area <= max_area * cluster_area_multiplier):
                    continue
            perimeter = cv2.arcLength(candidate, True)
            circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
            # 여러 개체가 뭉친 큰 덩어리는 낱개보다 원형도가 다소 떨어질 수 있으므로
            # 완전히 같은 기준을 적용하지 않고 문턱값을 낮춰준다.
            required_circularity = min_circularity * 0.6 if unresolved_cluster else min_circularity
            if circularity < required_circularity:
                continue
            m = cv2.moments(candidate)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            detections.append((cx, cy, area))

    detections.sort(key=lambda d: d[0])
    return detections, thresh


def detect_motion_objects(
    gray_current: np.ndarray,
    gray_previous: Optional[np.ndarray],
    roi_mask: Optional[np.ndarray],
    min_area: int = 8,
    max_area: int = 4000,
    sensitivity: int = 18,
) -> List[Tuple[float, float, float]]:
    """인접 프레임의 변화에서 실제로 움직인 개체 후보를 찾는다."""
    if gray_previous is None:
        return []
    if gray_current.shape != gray_previous.shape:
        gray_previous = cv2.resize(gray_previous, (gray_current.shape[1], gray_current.shape[0]))

    current = cv2.GaussianBlur(gray_current, (5, 5), 0)
    previous = cv2.GaussianBlur(gray_previous, (5, 5), 0)
    diff = cv2.absdiff(current, previous)
    _, motion = cv2.threshold(diff, int(sensitivity), 255, cv2.THRESH_BINARY)
    if roi_mask is not None:
        motion = cv2.bitwise_and(motion, motion, mask=roi_mask)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, open_kernel, iterations=1)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        detections.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], area))
    detections.sort(key=lambda detection: detection[0])
    return detections


def detect_contour_objects(
    frame_current: np.ndarray,
    roi_mask: Optional[np.ndarray],
    min_area: int = 15,
    max_area: int = 2000,
    sensitivity: int = 25,
    split_touching: bool = True,
    split_peak_ratio: float = 0.62,
    min_circularity: float = 0.20,
) -> Tuple[List[Tuple[float, float, float]], np.ndarray, List[np.ndarray]]:
    """현재 영상의 국부 대비에서 개체 외곽 컨투어를 직접 검출한다."""
    if frame_current.ndim == 3:
        gray_current = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_current, cv2.COLOR_BGR2HSV)
        warm_color = cv2.inRange(hsv, np.array([0, 20, 45]), np.array([45, 255, 255]))
    else:
        gray_current = frame_current
        warm_color = np.full_like(gray_current, 255)

    kernel_size = max(21, min(61, int(round(min(gray_current.shape[:2]) * 0.08))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    local_contrast = cv2.morphologyEx(gray_current, cv2.MORPH_TOPHAT, kernel)
    _, thresh = cv2.threshold(local_contrast, int(sensitivity), 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, warm_color)
    if roi_mask is not None:
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, clean_kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        candidates = (
            _split_touching_contour(
                contour, thresh.shape, min_area, 2.2, split_peak_ratio, 4
            )
            if split_touching
            else [contour]
        )
        for candidate in candidates:
            candidate_area = cv2.contourArea(candidate)
            if candidate_area < min_area or candidate_area > max_area:
                continue
            perimeter = cv2.arcLength(candidate, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * candidate_area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            moments = cv2.moments(candidate)
            if moments["m00"] == 0:
                continue
            detections.append(
                (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], candidate_area)
            )
            valid_contours.append(candidate)

    order = np.argsort([detection[0] for detection in detections])
    detections = [detections[index] for index in order]
    valid_contours = [valid_contours[index] for index in order]
    return detections, thresh, valid_contours


# ----------------------------------------------------------------
# 3. 중심점 기반 개체 추적기
# ----------------------------------------------------------------
@dataclass
class TrackedObject:
    object_id: int
    centroid: Tuple[float, float]
    first_seen: float
    last_seen: float
    last_move_time: Optional[float] = None
    missed_frames: int = 0
    motion_reference: Optional[Tuple[float, float]] = None
    total_distance: float = 0.0
    path: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)

    def __post_init__(self):
        if self.motion_reference is None:
            self.motion_reference = self.centroid
        if not self.path:
            self.path.append((self.first_seen, self.centroid[0], self.centroid[1]))


class CentroidTracker:
    """전역 최근접 중심점 매칭(그리디) 기반 다중개체 추적기.

    참고: Hungarian 알고리즘이 아닌 그리디(탐욕) 매칭을 사용하므로,
    개체가 서로 심하게 겹치는 순간에는 ID가 바뀔 수 있습니다.
    다만 모든 (트랙, 검출) 후보 쌍을 거리순으로 정렬한 뒤 전역으로 매칭하기 때문에,
    트랙별로 각자 가장 가까운 검출만 보던 이전 방식보다 스왑이 줄어듭니다.
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
            first_seen=timestamp, last_seen=timestamp, last_move_time=None,
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

        if len(detections) == 0:
            for oid in object_ids:
                self.objects[oid].missed_frames += 1
            self._deregister_stale()
            return self.objects

        # 모든 (트랙, 검출) 후보 쌍을 거리순으로 정렬한 뒤 전역으로 그리디 매칭한다.
        # (트랙별로 각자의 최단거리 검출만 보는 방식보다 근접/교차 상황에서 ID 스왑이 적다)
        candidate_pairs: List[Tuple[float, int, int]] = []
        for row, oid in enumerate(object_ids):
            ox, oy = self.objects[oid].centroid
            for col, (dx, dy) in enumerate(detections):
                dist = hypot(ox - dx, oy - dy)
                if dist <= self.max_distance:
                    candidate_pairs.append((dist, row, col))
        candidate_pairs.sort(key=lambda item: item[0])

        used_rows: set = set()
        used_cols: set = set()
        for dist, row, col in candidate_pairs:
            if row in used_rows or col in used_cols:
                continue
            oid = object_ids[row]
            self._apply_detection(self.objects[oid], detections[col], timestamp)
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

    def _apply_detection(self, obj: TrackedObject, new_centroid: Tuple[float, float], timestamp: float) -> None:
        frame_moved = float(np.hypot(new_centroid[0] - obj.centroid[0], new_centroid[1] - obj.centroid[1]))
        obj.total_distance += frame_moved

        # 직전 프레임과의 거리만 비교하면, 프레임당 변위가 임계값 미만인 느린 드리프트가
        # 누적돼도 절대 '이동'으로 판정되지 않는 사각지대가 생긴다. 대신 마지막으로 실제
        # 이동이 확정된 기준점(motion_reference)과 비교해 누적 변위로 판정한다.
        ref_x, ref_y = obj.motion_reference
        reference_moved = float(np.hypot(new_centroid[0] - ref_x, new_centroid[1] - ref_y))

        obj.centroid = new_centroid
        obj.last_seen = timestamp
        obj.missed_frames = 0
        obj.path.append((timestamp, new_centroid[0], new_centroid[1]))

        if reference_moved >= self.move_threshold_px:
            obj.last_move_time = timestamp
            obj.motion_reference = new_centroid

    def mark_as_moved(self, object_id: int, timestamp: float) -> bool:
        """수동으로 특정 개체의 '이동 확인' 시각을 초기화한다. (오탐 사멸의심 해제용)"""
        obj = self.objects.get(object_id)
        if obj is None:
            return False
        obj.last_move_time = timestamp
        obj.motion_reference = obj.centroid
        return True


# ----------------------------------------------------------------
# 4. 상태 분류
# ----------------------------------------------------------------
def classify_state(obj: TrackedObject, now: float, no_movement_threshold_sec: float,
                    observing_grace_sec: float = 5.0,
                    minimum_track_age_seconds: Optional[float] = None) -> str:
    """개체의 현재 상태를 분류한다.

    - 최근(활동 인정 시간 이내)에 실제로 움직인 개체        -> 활동 (갓 등장한 개체라도 즉시 반영)
    - 아직 나이가 어린(최소 추적 나이 이내) 개체            -> 관찰중 (성급한 사멸의심 방지)
    - 움직이지 않은 지 무이동 판정시간을 넘지 않은 개체      -> 관찰중
    - 무이동 판정시간을 넘긴 개체                          -> 사멸의심 (사망 확정 아님)

    이전 버전은 "나이가 어리면 무조건 관찰중"을 가장 먼저 체크했기 때문에, 등장하자마자
    실제로 움직인 개체까지 관찰 유예시간 동안 활동으로 인정받지 못하는 문제가 있었다.
    이제는 '최근 실제 이동 여부'를 나이와 무관하게 가장 먼저 확인하고, 최소 추적 나이는
    (아직 한 번도 움직이지 않은) 신규 개체가 너무 빨리 사멸의심으로 넘어가는 것을 막는
    별도의 안전장치로만 쓴다. minimum_track_age_seconds를 지정하지 않으면 기존과 동일하게
    observing_grace_sec 값을 그대로 사용한다.
    """
    if minimum_track_age_seconds is None:
        minimum_track_age_seconds = observing_grace_sec

    if obj.last_move_time is not None and now - obj.last_move_time < observing_grace_sec:
        return STATE_ACTIVE

    age = now - obj.first_seen
    if age < minimum_track_age_seconds:
        return STATE_OBSERVING

    inactive_since = obj.last_move_time if obj.last_move_time is not None else obj.first_seen
    if now - inactive_since < no_movement_threshold_sec:
        return STATE_OBSERVING
    return STATE_DEAD_SUSPECT
