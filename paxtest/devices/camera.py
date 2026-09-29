"""카메라 미리보기 + 녹화.

녹화 시 video.mp4 와 함께 video_frames.csv(frame, t_unix_s)를 남겨
영상 프레임을 힘 데이터와 정확히 대응시킬 수 있게 한다.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)
OverlayFn = Callable[[], List[str]]


class CameraBase(threading.Thread):
    def __init__(self, fps: float, overlay: bool = True) -> None:
        super().__init__(daemon=True)
        self.fps = fps
        self.overlay_enabled = overlay
        self.overlay_fn: Optional[OverlayFn] = None
        self.status = "disconnected"
        self.error = ""
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._rec_lock = threading.Lock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_file = None
        self._frame_idx = 0
        self._pending = None
        self._stop_evt = threading.Event()
        self.measured_fps = 0.0

    # ── 하위 클래스 구현 ──
    def _open(self) -> bool: ...
    def _read(self) -> Optional[np.ndarray]: ...
    def _close(self) -> None: ...

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    def start_recording(self, video_path: Path, frames_csv: Path) -> None:
        with self._rec_lock:
            self._pending = (Path(video_path), Path(frames_csv))

    def stop_recording(self) -> int:
        with self._rec_lock:
            self._pending = None
            n = self._frame_idx
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if self._frames_file is not None:
                self._frames_file.close()
                self._frames_file = None
            return n

    def stop(self) -> None:
        self._stop_evt.set()

    def _draw_overlay(self, frame: np.ndarray, t: float) -> None:
        lines = [time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"]
        if self.overlay_fn:
            try:
                lines += self.overlay_fn()
            except Exception:
                pass
        font, scale, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 24
        # cv2 는 한글 렌더링 불가 → overlay_fn 은 ASCII 만 반환
        w = max(cv2.getTextSize(s, font, scale, th)[0][0] for s in lines) + 20
        h = lh * len(lines) + 10
        roi = frame[0:min(h, frame.shape[0]), 0:min(w, frame.shape[1])]
        roi[:] = (roi * 0.35).astype(frame.dtype)        # 반투명 어두운 배경
        y = 22
        for s in lines:
            cv2.putText(frame, s, (10, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
            y += lh

    def _handle_recording(self, frame: np.ndarray, t: float) -> None:
        with self._rec_lock:
            pending = self._pending
            if pending and self._writer is None:
                video_path, frames_csv = pending
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self._frames_file = open(frames_csv, "w", encoding="utf-8")
                self._frames_file.write("frame,t_unix_s\n")
                self._frame_idx = 0
            if self._writer is not None:
                self._writer.write(frame)
                self._frames_file.write(f"{self._frame_idx},{t:.4f}\n")
                self._frame_idx += 1

    def run(self) -> None:
        if not self._open():
            self.status = "error"
            return
        self.status = "connected"
        stamps: List[float] = []
        while not self._stop_evt.is_set():
            frame = self._read()
            t = time.time()
            if frame is None:
                self._stop_evt.wait(0.05)
                continue
            if self.overlay_enabled:
                self._draw_overlay(frame, t)
            with self._frame_lock:
                self._frame = frame
            self._handle_recording(frame, t)
            stamps.append(t)
            stamps = stamps[-30:]
            if len(stamps) > 1:
                self.measured_fps = (len(stamps) - 1) / max(1e-6, stamps[-1] - stamps[0])
        self.stop_recording()
        self._close()
        self.status = "disconnected"


class OpenCVCamera(CameraBase):
    def __init__(self, cfg: dict) -> None:
        super().__init__(float(cfg.get("fps", 30)), bool(cfg.get("overlay", True)))
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> bool:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(int(self.cfg.get("index", 0)), backend)
        if not self.cap.isOpened():
            self.error = "카메라를 열 수 없음"
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get("width", 1280)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get("height", 720)))
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def _read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def _close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class SimCamera(CameraBase):
    """가상 장비용 합성 영상: 게이지 하중을 막대로 그린다."""

    def __init__(self, world, fps: float = 15, size=(640, 360)) -> None:
        super().__init__(fps, True)
        self.world = world
        self.w, self.h = size

    def _open(self) -> bool:
        self._next = time.time()
        return True

    def _read(self) -> Optional[np.ndarray]:
        now = time.time()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + 1.0 / self.fps, time.time())
        img = np.full((self.h, self.w, 3), (40, 38, 36), np.uint8)
        load = self.world.gauge_load(time.time())
        cv2.rectangle(img, (0, self.h - 70), (self.w, self.h), (60, 58, 56), -1)
        bar = int(min(1.0, max(0.0, load / 60.0)) * (self.w - 40))
        cv2.rectangle(img, (20, self.h - 50), (20 + bar, self.h - 25), (214, 120, 42), -1)
        cv2.putText(img, "SIM CAMERA", (self.w - 190, self.h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (160, 160, 160), 1, cv2.LINE_AA)
        return img

    def _close(self) -> None:
        pass
