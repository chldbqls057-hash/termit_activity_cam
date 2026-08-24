"""
termite_monitor_app.py
----------------------------------------------------------
흰개미 활동판독 - Windows 데스크톱 GUI 애플리케이션

Osmo Action(웹캠 모드)으로 촬영한 패트리디쉬(여과지) 영상에서
흰개미 개체를 자동 검출·추적하고, 활동/관찰중/사멸의심 상태를
색상으로 구분해 표시합니다. CSV 로그와 스크린샷/영상도 저장합니다.

실행:
    python termite_monitor_app.py

exe로 빌드하려면 build_exe.bat 를 참고하세요.
----------------------------------------------------------
"""

import csv
import os
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from termite_core import (
    STATE_COLORS,
    CentroidTracker,
    classify_state,
    detect_objects,
    detect_petri_dish,
    make_circular_mask,
)

APP_TITLE = "흰개미 활동판독"
DEFAULT_SAVE_ROOT = Path.home() / "Documents" / "TermiteMonitor_sessions"


class TermiteMonitorApp:
    # 패트리디쉬 테두리(그림자/반사가 생기기 쉬운 가장자리)를 검출 대상에서 살짝 제외
    ROI_INNER_RATIO = 0.95
    # 기준영상(빈 여과지) 촬영 시 잡음을 줄이기 위해 여러 프레임을 모아 중앙값을 사용
    REFERENCE_CAPTURE_FRAMES = 15
    # '분리 민감도' 콤보박스 -> watershed 피크 임계 비율. 낮을수록 더 쉽게 두 덩어리로 나눈다.
    SPLIT_SENSITIVITY_MAP = {"낮음": 0.56, "보통": 0.62, "높음": 0.69}
    # 신규 ID 발급 전 요구하는 시간축 안정성: 최근 몇 프레임을 볼지 / 그중 몇 번 이상
    # 근접 검출돼야 '진짜'로 인정할지. 반사·주름처럼 순간적으로 깜빡이는 오검출을 거른다.
    STABILITY_HISTORY_LEN = 4
    STABILITY_MIN_HITS = 2
    STABILITY_MATCH_RADIUS_PX = 25

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1180x760")
        root.minsize(980, 620)

        # --- 상태 변수 ---
        self.cap = None
        self.camera_index = tk.IntVar(value=0)
        self.running_analysis = False
        self.recording = False
        self.video_writer = None

        self.reference_gray = None
        self.roi_center = None
        self.roi_radius = None
        self.roi_mask = None

        self.tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)

        self.expected_count = tk.IntVar(value=0)  # 예상 투입 개체 수
        self.no_move_sec = tk.DoubleVar(value=300.0)  # 무이동 판정 시간(초), 기본 5분
        self.min_area = tk.IntVar(value=15)
        self.max_area = tk.IntVar(value=2000)
        self.sensitivity = tk.IntVar(value=25)
        self.min_circularity = tk.DoubleVar(value=0.35)
        self.log_interval_sec = 1.0

        self.session_dir = None
        self.positions_csv = None
        self.summary_csv = None
        self.last_log_time = 0.0

        self.manual_roi_mode = False
        self.manual_center_tmp = None

        self.last_frame_bgr = None
        self.canvas_scale = 1.0
        self.zoom = tk.DoubleVar(value=1.0)
        self.zoom_crop_origin = (0, 0)
        self.detection_history = []

        self._build_ui()
        self._poll_camera()

    # ------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="black", width=860, height=620)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        right = ttk.Frame(main, width=300)
        right.pack(side="right", fill="y", padx=(8, 0))

        cam_frame = ttk.LabelFrame(right, text="① 카메라")
        cam_frame.pack(fill="x", pady=4)
        self.camera_combo = ttk.Combobox(cam_frame, values=list(range(10)), textvariable=self.camera_index,
                                          width=5, state="readonly")
        self.camera_combo.pack(side="left", padx=4, pady=4)
        ttk.Button(cam_frame, text="연결", command=self.connect_camera).pack(side="left", padx=4)
        ttk.Button(cam_frame, text="목록 새로고침", command=self._refresh_camera_list).pack(side="left", padx=4)
        ttk.Label(cam_frame, text="확대").pack(side="left", padx=(8, 2))
        ttk.Scale(cam_frame, from_=1.0, to=3.0, variable=self.zoom, command=self._on_zoom_changed,
              length=100).pack(side="left", padx=2)
        ttk.Button(cam_frame, text="초기화", command=lambda: self.zoom.set(1.0)).pack(side="left", padx=2)

        roi_frame = ttk.LabelFrame(right, text="② 패트리디쉬 영역")
        roi_frame.pack(fill="x", pady=4)
        ttk.Button(roi_frame, text="자동 검출", command=self.auto_detect_dish).pack(fill="x", padx=4, pady=2)
        ttk.Button(roi_frame, text="수동 지정 (캔버스 클릭&드래그)", command=self.toggle_manual_roi).pack(
            fill="x", padx=4, pady=2)

        ref_frame = ttk.LabelFrame(right, text="③ 기준영상")
        ref_frame.pack(fill="x", pady=4)
        ttk.Button(ref_frame, text="빈 여과지 기준영상 촬영", command=self.capture_reference).pack(
            fill="x", padx=4, pady=2)

        param_frame = ttk.LabelFrame(right, text="④ 판독 파라미터")
        param_frame.pack(fill="x", pady=4)
        self._add_param_row(param_frame, "예상 투입 개체 수", self.expected_count, 0, 1000)
        self._add_param_row(param_frame, "무이동 판정(초)", self.no_move_sec, 10, 3600)
        self._add_param_row(param_frame, "최소 검출면적(px²)", self.min_area, 3, 500)
        self._add_param_row(param_frame, "최대 검출면적(px²)", self.max_area, 100, 20000)
        self._add_param_row(param_frame, "민감도(임계값)", self.sensitivity, 5, 100)
        self._add_param_row(param_frame, "최소 원형도(0~1)", self.min_circularity, 0.0, 1.0, increment=0.05)
        circularity_note = ttk.Label(
            param_frame,
            text="낮출수록 길쭉/구불구불한 모양도 개체로 인정합니다. 헝겊 주름·종이 테두리 같은"
                 " 이물질이 개체로 잘못 잡히면 값을 높이세요(둥근 개체만 인정).",
            wraplength=340, foreground="#555",
        )
        circularity_note.pack(fill="x", padx=4, pady=(0, 4))

        self.split_touching = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="겹친 개체 자동 분리", variable=self.split_touching).pack(
            anchor="w", padx=4, pady=(4, 0))
        split_row = ttk.Frame(param_frame)
        split_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(split_row, text="분리 민감도", width=16).pack(side="left")
        self.split_sensitivity = tk.StringVar(value="보통")
        ttk.Combobox(
            split_row, textvariable=self.split_sensitivity, width=8, state="readonly",
            values=["낮음", "보통", "높음"],
        ).pack(side="left")
        split_note = ttk.Label(
            param_frame,
            text="'낮음'일수록 뭉친 덩어리를 더 쉽게 나눕니다. 여전히 하나로 뭉쳐 보이면"
                 " 낮음으로, 낱개가 과하게 쪼개지면 높음으로 바꿔보세요.",
            wraplength=340, foreground="#555",
        )
        split_note.pack(fill="x", padx=4, pady=(0, 2))

        self.show_mask = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="검출 마스크 미리보기 표시", variable=self.show_mask).pack(
            anchor="w", padx=4, pady=(2, 4))

        run_frame = ttk.LabelFrame(right, text="⑤ 판독 제어")
        run_frame.pack(fill="x", pady=4)
        ttk.Button(run_frame, text="▶ 판독 시작", command=self.start_analysis).pack(fill="x", padx=4, pady=2)
        ttk.Button(run_frame, text="■ 판독 정지", command=self.stop_analysis).pack(fill="x", padx=4, pady=2)

        manual_frame = ttk.LabelFrame(right, text="수동 확인 (오탐 사멸의심 해제)")
        manual_frame.pack(fill="x", pady=4)
        manual_row = ttk.Frame(manual_frame)
        manual_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(manual_row, text="ID").pack(side="left")
        self.manual_track_id = tk.IntVar(value=0)
        ttk.Spinbox(manual_row, from_=0, to=9999, textvariable=self.manual_track_id, width=6).pack(
            side="left", padx=4)
        ttk.Button(manual_row, text="이동 확인", command=self.manual_mark_moved).pack(side="left", padx=2)
        ttk.Button(manual_frame, text="전체 ID 초기화", command=self.reset_tracks).pack(fill="x", padx=4, pady=2)

        save_frame = ttk.LabelFrame(right, text="⑥ 저장")
        save_frame.pack(fill="x", pady=4)
        ttk.Button(save_frame, text="스크린샷 저장", command=self.save_screenshot).pack(fill="x", padx=4, pady=2)
        ttk.Button(save_frame, text="영상 녹화 시작/정지", command=self.toggle_recording).pack(fill="x", padx=4, pady=2)
        ttk.Button(save_frame, text="저장 폴더 열기", command=self.open_session_folder).pack(fill="x", padx=4, pady=2)

        status_frame = ttk.LabelFrame(right, text="현재 상태")
        status_frame.pack(fill="x", pady=4)
        self.lbl_active = ttk.Label(status_frame, text="활동: 0", foreground="#1a7d1a")
        self.lbl_active.pack(anchor="w", padx=4)
        self.lbl_observing = ttk.Label(status_frame, text="관찰중: 0", foreground="#b8860b")
        self.lbl_observing.pack(anchor="w", padx=4)
        self.lbl_dead = ttk.Label(status_frame, text="사멸의심: 0", foreground="#b30000")
        self.lbl_dead.pack(anchor="w", padx=4)
        self.lbl_total = ttk.Label(status_frame, text="총 개체수: 0")
        self.lbl_total.pack(anchor="w", padx=4)
        self.lbl_mismatch = ttk.Label(status_frame, text="", foreground="#b30000", wraplength=340)
        self.lbl_mismatch.pack(anchor="w", padx=4, pady=(2, 0))

        log_frame = ttk.LabelFrame(self.root, text="로그")
        log_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.log_box = tk.Text(log_frame, height=6)
        self.log_box.pack(fill="x")
        self._log("프로그램을 시작했습니다. ①카메라 연결 → ②영역 지정 → ③기준영상 촬영 → ⑤판독 시작 순서로 진행하세요.")

    def _add_param_row(self, parent, label, var, frm, to, increment=1):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Spinbox(row, from_=frm, to=to, increment=increment, textvariable=var, width=8).pack(side="left")

    # ------------------------------------------------------------
    # 카메라
    # ------------------------------------------------------------
    def _refresh_camera_list(self):
        # 스마트폰(아이폰/갤럭시)을 가상 웹캠 앱으로 연결하면 보통 기존 내장/USB 카메라보다
        # 뒤쪽 인덱스에 잡히므로 넉넉히 0~9까지 훑는다.
        found = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
            cap.release()
        if not found:
            found = [0]
        self.camera_combo.config(values=found)
        self._log(f"감지된 카메라 인덱스: {found}"
                  " (스마트폰은 DroidCam/EpocCam/Camo/Iriun 같은 가상 웹캠 앱을 설치·실행해야 목록에 나타납니다)")

    def connect_camera(self):
        idx = self.camera_index.get()
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)  # 일부 가상 웹캠 드라이버는 DSHOW 대신 이쪽에서만 열림
        if not cap.isOpened():
            messagebox.showerror(
                APP_TITLE,
                f"카메라(index={idx})를 열 수 없습니다.\n"
                "- Osmo라면 '웹캠' 모드인지 확인하세요.\n"
                "- 스마트폰(아이폰/갤럭시)은 그 자체로 Windows 표준 웹캠으로 인식되지 않습니다.\n"
                "  갤럭시(안드로이드)는 DroidCam이나 Iriun Webcam,\n"
                "  아이폰은 EpocCam(Kinoni)이나 Camo(Reincubate), Iriun Webcam 같은 앱을\n"
                "  PC와 휴대폰 양쪽에 설치해 가상 웹캠으로 연결한 뒤,\n"
                "  '목록 새로고침'으로 해당 카메라 인덱스를 찾아 선택하세요.",
            )
            return
        self.cap = cap
        self._log(f"카메라 index {idx} 연결됨")

    def _on_zoom_changed(self, _value=None):
        if self.last_frame_bgr is not None:
            self._render_canvas(self.last_frame_bgr)

    # ------------------------------------------------------------
    # 영역 지정
    # ------------------------------------------------------------
    def auto_detect_dish(self):
        if self.last_frame_bgr is None:
            messagebox.showwarning(APP_TITLE, "먼저 카메라를 연결하세요.")
            return
        gray = cv2.cvtColor(self.last_frame_bgr, cv2.COLOR_BGR2GRAY)
        result = detect_petri_dish(gray)
        if result is None:
            messagebox.showinfo(APP_TITLE, "자동 검출에 실패했습니다. '수동 지정'을 사용하세요.")
            return
        cx, cy, r = result
        self.roi_center = (cx, cy)
        self.roi_radius = r
        self.roi_mask = make_circular_mask(self.last_frame_bgr.shape, cx, cy, r, inner_ratio=self.ROI_INNER_RATIO)
        self._log(f"패트리디쉬 자동 검출: 중심=({cx},{cy}), 반지름={r}px")

    def toggle_manual_roi(self):
        self.manual_roi_mode = not self.manual_roi_mode
        state = "켜짐 (영상 위를 클릭한 채 드래그해서 원을 그리세요)" if self.manual_roi_mode else "꺼짐"
        self._log(f"수동 영역 지정 모드: {state}")

    def _on_canvas_press(self, event):
        if not self.manual_roi_mode:
            return
        self.manual_center_tmp = (
            event.x / self.canvas_scale + self.zoom_crop_origin[0],
            event.y / self.canvas_scale + self.zoom_crop_origin[1],
        )

    def _on_canvas_drag(self, event):
        if not self.manual_roi_mode or self.manual_center_tmp is None:
            return
        cx, cy = self.manual_center_tmp
        x = event.x / self.canvas_scale + self.zoom_crop_origin[0]
        y = event.y / self.canvas_scale + self.zoom_crop_origin[1]
        dx = x - cx
        dy = y - cy
        r = int((dx ** 2 + dy ** 2) ** 0.5)
        self.roi_center = (int(cx), int(cy))
        self.roi_radius = max(r, 5)

    def _on_canvas_release(self, event):
        if not self.manual_roi_mode or self.roi_center is None or self.roi_radius is None:
            return
        if self.last_frame_bgr is not None:
            self.roi_mask = make_circular_mask(
                self.last_frame_bgr.shape, self.roi_center[0], self.roi_center[1], self.roi_radius,
                inner_ratio=self.ROI_INNER_RATIO)
        self._log(f"수동 영역 지정 완료: 중심={self.roi_center}, 반지름={self.roi_radius}px")
        self.manual_roi_mode = False
        self.manual_center_tmp = None

    # ------------------------------------------------------------
    # 기준영상
    # ------------------------------------------------------------
    def capture_reference(self):
        if self.cap is None or not self.cap.isOpened():
            messagebox.showwarning(APP_TITLE, "먼저 카메라를 연결하세요.")
            return
        # 프레임 1장만 쓰면 카메라 센서 잡음이 그대로 기준영상에 남아 오검출로 이어지기 쉽다.
        # 짧게 여러 프레임을 모아 중앙값을 취해 잡음을 줄인다.
        frames = []
        for _ in range(self.REFERENCE_CAPTURE_FRAMES):
            ok, frame = self.cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if not frames:
            if self.last_frame_bgr is None:
                messagebox.showwarning(APP_TITLE, "카메라 프레임을 읽지 못했습니다.")
                return
            frames = [cv2.cvtColor(self.last_frame_bgr, cv2.COLOR_BGR2GRAY)]
        self.reference_gray = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
        self._log(f"기준영상(빈 여과지) 촬영 완료 ({len(frames)}프레임 중앙값). "
                  "이제 흰개미를 투입한 뒤 '판독 시작'을 누르세요.")

    # ------------------------------------------------------------
    # 판독 제어
    # ------------------------------------------------------------
    def start_session_folder(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = DEFAULT_SAVE_ROOT / f"session_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.positions_csv = self.session_dir / "positions_log.csv"
        self.summary_csv = self.session_dir / "summary_log.csv"
        with open(self.positions_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["timestamp", "id", "x", "y", "state"])
        with open(self.summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["timestamp", "active", "observing", "dead_suspect", "total"])
        self._log(f"세션 폴더 생성: {self.session_dir}")

    def start_analysis(self):
        if self.cap is None or not self.cap.isOpened():
            messagebox.showwarning(APP_TITLE, "먼저 카메라를 연결하세요.")
            return
        if self.reference_gray is None:
            messagebox.showwarning(APP_TITLE, "먼저 기준영상(빈 여과지)을 촬영하세요.")
            return
        if self.roi_mask is None:
            if not messagebox.askyesno(APP_TITLE, "패트리디쉬 영역이 지정되지 않았습니다. 화면 전체를 대상으로 진행할까요?"):
                return
        try:
            self.start_session_folder()
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"세션 폴더 생성 실패: {e}")
            return
        self.tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)
        self.detection_history = []
        self.running_analysis = True
        self.last_log_time = 0.0
        self._log("판독을 시작합니다.")

    def stop_analysis(self):
        self.running_analysis = False
        self._log("판독을 정지했습니다.")

    def manual_mark_moved(self):
        oid = self.manual_track_id.get()
        if self.tracker.mark_as_moved(oid, time.time()):
            self._log(f"ID {oid}의 이동 기준 시각을 초기화했습니다.")
        else:
            messagebox.showwarning(APP_TITLE, f"ID {oid}를 찾을 수 없습니다.")

    def reset_tracks(self):
        self.tracker = CentroidTracker(max_distance=60, max_missed_frames=20, move_threshold_px=3)
        self.detection_history = []
        self._log("모든 개체 ID를 초기화했습니다.")

    # ------------------------------------------------------------
    # 저장 (스크린샷 / 영상 / 폴더 열기)
    # ------------------------------------------------------------
    def toggle_recording(self):
        if not self.recording:
            if self.session_dir is None:
                messagebox.showwarning(APP_TITLE, "판독을 먼저 시작하세요 (세션 폴더가 필요합니다).")
                return
            if self.last_frame_bgr is None:
                messagebox.showwarning(APP_TITLE, "영상 프레임이 아직 없습니다.")
                return
            path = self.session_dir / f"recording_{datetime.now().strftime('%H%M%S')}.mp4"
            h, w = self.last_frame_bgr.shape[:2]
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(str(path), fourcc, 15.0, (w, h))
                self.recording = True
                self._log(f"영상 녹화 시작: {path}")
            except Exception as e:
                messagebox.showerror(APP_TITLE, f"녹화 시작 실패: {e}")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self._log("영상 녹화 정지")

    def save_screenshot(self):
        if self.last_frame_bgr is None:
            messagebox.showwarning(APP_TITLE, "저장할 영상 프레임이 없습니다.")
            return
        folder = self.session_dir if self.session_dir else DEFAULT_SAVE_ROOT
        folder.mkdir(parents=True, exist_ok=True)
        path = Path(folder) / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), self.last_frame_bgr)
        self._log(f"스크린샷 저장: {path}")

    def open_session_folder(self):
        folder = self.session_dir if self.session_dir else DEFAULT_SAVE_ROOT
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # Windows 전용
        except Exception:
            self._log(f"저장 폴더 위치: {folder}")

    # ------------------------------------------------------------
    # 메인 처리 루프
    # ------------------------------------------------------------
    def _poll_camera(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok:
                    self._process_frame(frame)
                else:
                    self._log("[경고] 프레임을 읽지 못했습니다. 케이블/연결을 확인하세요.")
        except Exception as e:
            self._log(f"[오류] {e}")
        self.root.after(33, self._poll_camera)

    def _process_frame(self, frame_bgr):
        display = frame_bgr.copy()
        now = time.time()

        if self.roi_center and self.roi_radius:
            cv2.circle(display, self.roi_center, self.roi_radius, (255, 200, 0), 2)

        counts = {"활동": 0, "관찰중": 0, "사멸의심": 0}

        mask = None
        if self.running_analysis and self.reference_gray is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detections, mask = detect_objects(
                gray, self.reference_gray, self.roi_mask,
                min_area=self.min_area.get(), max_area=self.max_area.get(),
                sensitivity=self.sensitivity.get(),
                split_touching=self.split_touching.get(),
                split_peak_ratio=self.SPLIT_SENSITIVITY_MAP.get(self.split_sensitivity.get(), 0.62),
                min_circularity=self.min_circularity.get(),
            )
            detections = self._keep_stable_detections(detections)
            centroids = [(d[0], d[1]) for d in detections]
            objects = self.tracker.update(centroids, now)

            for oid, obj in objects.items():
                state = classify_state(obj, now, self.no_move_sec.get())
                counts[state] = counts.get(state, 0) + 1
                color = STATE_COLORS[state]
                x, y = int(obj.centroid[0]), int(obj.centroid[1])
                cv2.circle(display, (x, y), 6, color, 2)
                cv2.putText(display, f"#{oid}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            if now - self.last_log_time >= self.log_interval_sec:
                self._write_logs(objects, now, counts)
                self.last_log_time = now

            self._update_status_labels(counts)
            self._update_mismatch_warning(sum(counts.values()))

        if self.show_mask.get() and mask is not None:
            display = self._embed_mask_preview(display, mask)

        self.last_frame_bgr = display
        self._render_canvas(display)

        if self.recording and self.video_writer is not None:
            self.video_writer.write(display)

    def _keep_stable_detections(self, detections):
        """최근 몇 프레임 중 일정 횟수 이상 같은 자리에 나타난 검출만 통과시킨다.

        반사·주름처럼 순간적으로 한두 프레임만 반짝이는 오검출은 시간축에서 걸러지고,
        실제로 존재하는 흰개미(연속으로 검출됨)는 몇 프레임(약 0.1초) 지연 후 통과한다.
        """
        radius_sq = self.STABILITY_MATCH_RADIUS_PX ** 2
        current_centroids = [(d[0], d[1]) for d in detections]
        stable = []
        for detection in detections:
            hits = sum(
                1
                for previous_centroids in self.detection_history
                if any(
                    (detection[0] - x) ** 2 + (detection[1] - y) ** 2 <= radius_sq
                    for x, y in previous_centroids
                )
            )
            if hits >= self.STABILITY_MIN_HITS:
                stable.append(detection)
        self.detection_history.append(current_centroids)
        self.detection_history = self.detection_history[-(self.STABILITY_HISTORY_LEN - 1):]
        return stable

    def _write_logs(self, objects, now, counts):
        if not self.positions_csv:
            return
        ts = datetime.now().isoformat(timespec="seconds")
        try:
            with open(self.positions_csv, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                for oid, obj in objects.items():
                    state = classify_state(obj, now, self.no_move_sec.get())
                    w.writerow([ts, oid, f"{obj.centroid[0]:.1f}", f"{obj.centroid[1]:.1f}", state])
            with open(self.summary_csv, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    [ts, counts.get("활동", 0), counts.get("관찰중", 0), counts.get("사멸의심", 0), sum(counts.values())])
        except Exception as e:
            self._log(f"[오류] 로그 저장 실패: {e}")

    def _render_canvas(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        canvas_w = self.canvas.winfo_width() or 860
        canvas_h = self.canvas.winfo_height() or 620
        scale = max(min(canvas_w / w, canvas_h / h), 0.1) * self.zoom.get()
        crop_w = min(w, max(int(canvas_w / scale), 1))
        crop_h = min(h, max(int(canvas_h / scale), 1))
        crop_x = max((w - crop_w) // 2, 0)
        crop_y = max((h - crop_h) // 2, 0)
        self.zoom_crop_origin = (crop_x, crop_y)
        self.canvas_scale = scale
        cropped = frame_bgr[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        disp = cv2.resize(cropped, (max(int(crop_w * scale), 1), max(int(crop_h * scale), 1)))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def _update_status_labels(self, counts):
        self.lbl_active.config(text=f"활동: {counts.get('활동', 0)}")
        self.lbl_observing.config(text=f"관찰중: {counts.get('관찰중', 0)}")
        self.lbl_dead.config(text=f"사멸의심: {counts.get('사멸의심', 0)}")
        self.lbl_total.config(text=f"총 개체수: {sum(counts.values())}")

    def _update_mismatch_warning(self, detected_count: int) -> None:
        expected = self.expected_count.get()
        if expected > 0 and detected_count != expected:
            self.lbl_mismatch.config(
                text=f"⚠ 검출 {detected_count}개 / 예상 투입 {expected}개 불일치 "
                     "— 겹친 개체가 있을 수 있습니다. '검출 마스크 미리보기'로 확인 후 "
                     "분리 민감도·최소/최대 면적을 조정하세요."
            )
        else:
            self.lbl_mismatch.config(text="")

    @staticmethod
    def _embed_mask_preview(frame_bgr, mask):
        """검출 이진 마스크를 화면 우하단에 작게 겹쳐 보여준다 (파라미터 튜닝용)."""
        output = frame_bgr.copy()
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        width = max(160, frame_bgr.shape[1] // 4)
        height = int(round(mask.shape[0] * width / mask.shape[1]))
        mask_bgr = cv2.resize(mask_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
        x = frame_bgr.shape[1] - width - 12
        y = frame_bgr.shape[0] - height - 12
        cv2.rectangle(output, (x - 2, y - 2), (x + width + 2, y + height + 2), (255, 255, 255), 2)
        output[y:y + height, x:x + width] = mask_bgr
        return output

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")

    def on_close(self):
        self.running_analysis = False
        if self.recording and self.video_writer:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TermiteMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()