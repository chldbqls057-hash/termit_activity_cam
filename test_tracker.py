"""
test_tracker.py
----------------------------------------------------------
GUI 없이 termite_core.py의 핵심 로직을 검증하는 합성(가상) 테스트 6종.
개발자가 로직을 수정할 때마다:  python test_tracker.py
로 재실행해서 회귀 여부를 확인할 수 있습니다.
----------------------------------------------------------
"""

import sys

import cv2
import numpy as np

from termite_core import (
    STATE_ACTIVE,
    STATE_DEAD_SUSPECT,
    STATE_OBSERVING,
    CentroidTracker,
    classify_state,
    detect_objects,
    detect_petri_dish,
)

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


# ------------------------------------------------------------
# 시나리오 1: 지속적으로 이동하는 단일 개체는 '활동' 상태를 유지해야 함
# ------------------------------------------------------------
def test_1_continuous_movement_stays_active():
    print("\n[시나리오 1] 지속 이동 개체 -> '활동' 상태 유지")
    tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)
    t = 0.0
    pos = (50.0, 50.0)
    objects = tracker.update([pos], t)
    oid = list(objects.keys())[0]

    for i in range(5):
        t += 1.0
        pos = (pos[0] + 10, pos[1])  # 계속 이동
        objects = tracker.update([pos], t)

    obj = tracker.objects[oid]
    state = classify_state(obj, t, no_movement_threshold_sec=300, observing_grace_sec=2.0)
    check("지속 이동 시 활동 상태", state == STATE_ACTIVE, f"실제 상태={state}")


# ------------------------------------------------------------
# 시나리오 2: 정지한 개체는 무이동 판정시간 초과 후 '사멸의심'으로 전환
# ------------------------------------------------------------
def test_2_stationary_object_becomes_dead_suspect():
    print("\n[시나리오 2] 정지 개체 -> 무이동시간 초과 후 '사멸의심'")
    tracker = CentroidTracker(max_distance=60, max_missed_frames=50, move_threshold_px=3)
    t = 0.0
    pos = (100.0, 100.0)
    objects = tracker.update([pos], t)
    oid = list(objects.keys())[0]

    no_move_threshold = 5.0
    grace = 1.0

    # 정지 상태로 계속 같은 위치에서 검출됨
    for i in range(10):
        t += 1.0
        objects = tracker.update([pos], t)  # 움직이지 않음

    obj = tracker.objects[oid]
    state_before = classify_state(obj, now=2.0, no_movement_threshold_sec=no_move_threshold, observing_grace_sec=grace)
    state_after = classify_state(obj, now=t, no_movement_threshold_sec=no_move_threshold, observing_grace_sec=grace)

    check("무이동시간 초과 전에는 사멸의심이 아님", state_before != STATE_DEAD_SUSPECT, f"state_before={state_before}")
    check("무이동시간 초과 후 사멸의심 전환", state_after == STATE_DEAD_SUSPECT, f"state_after={state_after}")


# ------------------------------------------------------------
# 시나리오 3: 새로운 개체 등장 시 새 ID 부여
# ------------------------------------------------------------
def test_3_new_object_gets_new_id():
    print("\n[시나리오 3] 신규 개체 등장 -> 새로운 ID 부여")
    tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)
    t = 0.0
    objects = tracker.update([(30.0, 30.0)], t)
    first_id = list(objects.keys())[0]

    t += 1.0
    # 기존 개체 근처 + 멀리 떨어진 새 개체 동시 검출
    objects = tracker.update([(35.0, 30.0), (300.0, 300.0)], t)

    check("객체 수 2개로 증가", len(objects) == 2, f"len={len(objects)}")
    check("기존 ID 유지됨", first_id in objects, f"objects keys={list(objects.keys())}")
    new_ids = [oid for oid in objects if oid != first_id]
    check("신규 개체에 새 ID 부여", len(new_ids) == 1)


# ------------------------------------------------------------
# 시나리오 4: 두 개체가 근접/겹칠 때도 추적기가 예외 없이 동작해야 함
# ------------------------------------------------------------
def test_4_close_objects_no_crash_and_recover():
    print("\n[시나리오 4] 근접/겹침 개체 -> 예외 없이 동작 + 겹침 이후 개체수 복구")
    tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)
    t = 0.0
    objects = tracker.update([(100.0, 100.0), (140.0, 100.0)], t)
    check("초기 2개체 등록", len(objects) == 2)

    # 서로 가까워지다가 겹쳐서 하나의 덩어리(검출 1개)로 인식되는 프레임 시뮬레이션
    for i in range(3):
        t += 1.0
        objects = tracker.update([(118.0, 100.0)], t)  # 겹쳐서 1개만 검출됨

    # 다시 분리되어 2개로 검출
    t += 1.0
    try:
        objects = tracker.update([(105.0, 100.0), (135.0, 100.0)], t)
        crashed = False
    except Exception as e:
        crashed = True
        print(f"    예외 발생: {e}")

    check("겹침 상황에서 예외 발생하지 않음", not crashed)
    check("분리 이후 개체수 2로 복구", len(objects) == 2, f"len={len(objects)}")


# ------------------------------------------------------------
# 시나리오 5: 일시적으로 가려져 검출 실패 후 재검출 시 같은 ID 유지
# ------------------------------------------------------------
def test_5_temporary_occlusion_keeps_id():
    print("\n[시나리오 5] 일시적 가림(검출 실패) 후 재검출 -> 동일 ID 유지")
    tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)
    t = 0.0
    objects = tracker.update([(200.0, 200.0)], t)
    oid = list(objects.keys())[0]

    # 3프레임 동안 검출 실패 (가림)
    for i in range(3):
        t += 1.0
        objects = tracker.update([], t)

    check("가려진 동안에도 ID 유지(제거되지 않음)", oid in tracker.objects)

    # 비슷한 위치에서 재검출
    t += 1.0
    objects = tracker.update([(203.0, 202.0)], t)

    check("재검출 시 동일 ID로 매칭됨", oid in objects and len(objects) == 1, f"objects={list(objects.keys())}")


# ------------------------------------------------------------
# 시나리오 6: 다중 개체(5마리) 동시 추적 시 개체수 정확성
# ------------------------------------------------------------
def test_6_multi_object_count_accuracy():
    print("\n[시나리오 6] 다중 개체(5마리) 동시 추적 -> 개체수 정확성")
    tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)
    t = 0.0
    initial_positions = [(50.0 * i, 50.0) for i in range(1, 6)]  # 5마리, 서로 충분히 떨어짐
    objects = tracker.update(initial_positions, t)
    check("5마리 최초 등록", len(objects) == 5, f"len={len(objects)}")

    # 각자 조금씩 다른 방향으로 5프레임 이동
    for i in range(5):
        t += 1.0
        positions = [(x + (i % 3), y + 1) for (x, y) in initial_positions]
        initial_positions = positions
        objects = tracker.update(positions, t)

    check("이동 후에도 5마리 유지", len(objects) == 5, f"len={len(objects)}")
    check("ID 중복 없음", len(set(objects.keys())) == len(objects))


# ------------------------------------------------------------
# 보너스: 실제 이미지 파이프라인(합성 패트리디쉬 영상) 테스트
# 배경차분 기반 검출 + 원(패트리디쉬) 검출이 실제로 동작하는지 확인
# ------------------------------------------------------------
def test_7_image_pipeline_on_synthetic_petri_dish():
    print("\n[보너스] 합성 패트리디쉬 영상 -> 원 검출 + 개체(흰개미) 검출 파이프라인")
    size = 300
    bg_value = 200

    # 기준영상: 빈 여과지 (패트리디쉬 원 + 균일한 배경)
    reference = np.full((size, size), bg_value, dtype=np.uint8)
    cv2.circle(reference, (size // 2, size // 2), 120, 150, thickness=3)  # 접시 테두리
    reference = cv2.GaussianBlur(reference, (5, 5), 0)

    dish = detect_petri_dish(reference)
    check("패트리디쉬(원) 자동 검출 성공", dish is not None)
    if dish:
        cx, cy, r = dish
        check("검출된 중심이 실제 중심과 근접", abs(cx - size // 2) < 20 and abs(cy - size // 2) < 20,
              f"검출값=({cx},{cy},{r})")
        check("검출된 반지름이 실제와 근접(±25px)", abs(r - 120) < 25, f"검출 반지름={r}")

    # 현재영상: 기준영상 + 어두운 점 2개(흰개미로 가정)를 추가
    current = reference.copy()
    cv2.circle(current, (140, 150), 5, 80, thickness=-1)
    cv2.circle(current, (170, 130), 4, 80, thickness=-1)

    detections, _ = detect_objects(current, reference, roi_mask=None, min_area=5, max_area=500, sensitivity=25)
    check("합성 개체 2마리 검출", len(detections) == 2, f"검출 개수={len(detections)}, 상세={detections}")


def main():
    tests = [
        test_1_continuous_movement_stays_active,
        test_2_stationary_object_becomes_dead_suspect,
        test_3_new_object_gets_new_id,
        test_4_close_objects_no_crash_and_recover,
        test_5_temporary_occlusion_keeps_id,
        test_6_multi_object_count_accuracy,
        test_7_image_pipeline_on_synthetic_petri_dish,
    ]
    for fn in tests:
        fn()

    print("\n" + "=" * 50)
    print(f"통과: {len(PASS)}건 / 실패: {len(FAIL)}건")
    if FAIL:
        print("실패한 항목:", FAIL)
        sys.exit(1)
    else:
        print("모든 테스트 통과!")
        sys.exit(0)


if __name__ == "__main__":
    main()
