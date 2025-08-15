import sys
import sys
import math
import time
from typing import Optional

# NOTE TO CONTRIBUTORS: QPointF is provided by PyQt6.QtCore on some builds;
#       QPolygonF stays in QtGui. Import accordingly.
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSlider, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QFont, QPolygonF

try:
    from moviepy.editor import VideoFileClip
except Exception:
    try:
        from moviepy import VideoFileClip
    except Exception:
        VideoFileClip = None  # will handle later

import numpy as np
from PIL import Image

# Pillow compatibility for resampling name
if hasattr(Image, "Resampling"):
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
else:
    RESAMPLE_BILINEAR = Image.BILINEAR

# ---------- Utility functions ----------

def sec_to_hms(t: float) -> str:
    if t is None:
        return "0:00"
    t = max(0.0, float(t))
    hrs = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int(t % 60)
    if hrs > 0:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"

def frame_to_qpixmap(frame: np.ndarray, target_w: int, target_h: int) -> QPixmap:

    try:
        img = Image.fromarray(frame)
        img = img.resize((max(1, target_w), max(1, target_h)), RESAMPLE_BILINEAR)
        img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
    except Exception:
        pm = QPixmap(max(1, target_w), max(1, target_h))
        pm.fill(QColor("#000000"))
        return pm

# ---------- Timeline Widget (custom-drawn) ----------

class TimelineWidget(QWidget):

    range_changed = pyqtSignal(float, float)
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # timeline data
        self.clip_duration = 0.0  # length of clip shown (seconds)
        self.playhead = 0.0       # current time (seconds)
        self.start_marker = 0.0
        self.end_marker = 0.0

        # visuals
        self.left_padding = 50
        self.right_padding = 50
        self.top_padding = 10
        self.bottom_padding = 10

        # dragging state
        self.dragging = None  # "start", "end", "playhead", or None
        self.drag_offset = 0.0

        # playback scale (zoom)
        self.pixels_per_second = 120.0  # default scale; adjustable
        self.min_pps = 40.0
        self.max_pps = 1000.0

        # display tick marks
        self.tick_height = 10

    def set_clip_duration(self, duration: float):
        self.clip_duration = float(max(0.0, duration))
        self.start_marker = 0.0
        self.end_marker = self.clip_duration
        self.playhead = 0.0
        self.update()

    def seconds_to_x(self, seconds: float) -> float:
        usable = max(10, self.width() - (self.left_padding + self.right_padding))
        total_width = self.pixels_per_second * max(1.0, self.clip_duration)
        # If total_width smaller than usable, we center, otherwise we use scaling so that full duration fits
        if total_width <= usable:
            # center the clip box
            start_x = self.left_padding + (usable - total_width) / 2
            return start_x + seconds * self.pixels_per_second
        else:
            # viewport shows full duration scaled to usable width
            scale = usable / (self.clip_duration if self.clip_duration > 0 else 1)
            start_x = self.left_padding
            return start_x + seconds * scale

    def x_to_seconds(self, x: float) -> float:
        usable = max(10, self.width() - (self.left_padding + self.right_padding))
        total_width = self.pixels_per_second * max(1.0, self.clip_duration)
        if total_width <= usable:
            start_x = self.left_padding + (usable - total_width) / 2
            return max(0.0, min(self.clip_duration, (x - start_x) / max(1e-6, self.pixels_per_second)))
        else:
            start_x = self.left_padding
            scale = usable / (self.clip_duration if self.clip_duration > 0 else 1)
            return max(0.0, min(self.clip_duration, (x - start_x) / max(1e-6, scale)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        # background
        painter.fillRect(0, 0, w, h, QColor("#2b2b2b"))

        usable_x = self.left_padding
        usable_w = max(10, w - (self.left_padding + self.right_padding))
        usable_y = self.top_padding
        usable_h = h - (self.top_padding + self.bottom_padding)

        # draw clip box background
        clip_x = usable_x
        clip_y = usable_y + 20
        clip_h = usable_h - 40
        # compute actual clip rectangle width
        total_width = self.pixels_per_second * max(1.0, self.clip_duration)
        if total_width <= usable_w:
            clip_w = int(total_width)
            clip_x = usable_x + (usable_w - clip_w) / 2
        else:
            clip_w = usable_w

        clip_rect = QRectF(clip_x, clip_y, clip_w, clip_h)
        painter.setBrush(QBrush(QColor("#444444")))
        painter.setPen(QPen(QColor("#555555")))
        painter.drawRoundedRect(clip_rect, 4, 4)

        # draw clip thumbnail-like stripes (visual interest)
        stripe_w = 8
        x = clip_x
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#3a8fb0")))
        while x < clip_x + clip_w:
            painter.drawRect(int(x), int(clip_y + 2), int(min(stripe_w - 2, clip_x + clip_w - x)), int(clip_h - 4))
            x += stripe_w + 10

        # draw start/end markers
        start_x = self.seconds_to_x(self.start_marker)
        end_x = self.seconds_to_x(self.end_marker)
        # range shaded selection
        sel_x = start_x
        sel_w = max(2, end_x - start_x)
        selection_rect = QRectF(sel_x, clip_y, sel_w, clip_h)
        painter.setBrush(QBrush(QColor(0, 160, 255, 90)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(selection_rect)

        # markers (vertical lines)
        painter.setPen(QPen(QColor("#00c0ff"), 2))
        painter.drawLine(int(start_x), clip_y, int(start_x), clip_y + clip_h)
        painter.drawLine(int(end_x), clip_y, int(end_x), clip_y + clip_h)

        # draw draggable triangular handles for markers
        tri_h = 12
        painter.setBrush(QBrush(QColor("#00c0ff")))
        # prepare triangle points as QPointF lists
        start_tri = [QPointF(start_x - 6, clip_y - 2), QPointF(start_x + 6, clip_y - 2), QPointF(start_x, clip_y - 2 - tri_h)]
        end_tri = [QPointF(end_x - 6, clip_y + clip_h + 2 + tri_h), QPointF(end_x + 6, clip_y + clip_h + 2 + tri_h), QPointF(end_x, clip_y + clip_h + 2)]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF(start_tri))
        painter.drawPolygon(QPolygonF(end_tri))

        # draw playhead (red)
        ph_x = self.seconds_to_x(self.playhead)
        painter.setPen(QPen(QColor("#ff5555"), 2))
        painter.drawLine(int(ph_x), clip_y - 6, int(ph_x), clip_y + clip_h + 6)

        # draw time ticks & labels along bottom
        tick_y = clip_y + clip_h + 10
        painter.setPen(QPen(QColor("#cccccc"), 1))
        # choose tick step based on duration
        duration = max(1.0, self.clip_duration)
        usable_clip_w = clip_w
        # choose seconds per tick so that ticks are roughly ~80-160 px apart
        target_px = 120.0
        seconds_per_tick = max(1.0, target_px / max(1e-6, (usable_clip_w / duration)))
        # round to nice value
        nice = [1,2,5,10,15,30,60,120,300,600]
        seconds_per_tick = min(nice, key=lambda x: abs(x - seconds_per_tick))
        # draw ticks
        t = 0.0
        font = QFont("Arial", 8)
        painter.setFont(font)
        while t <= duration + 0.0001:
            tx = self.seconds_to_x(t)
            painter.drawLine(int(tx), int(tick_y), int(tx), int(tick_y + self.tick_height))
            label = sec_to_hms(t)
            painter.drawText(int(tx - 20), int(tick_y + self.tick_height + 14), 40, 16, Qt.AlignmentFlag.AlignCenter, label)
            t += seconds_per_tick

        # draw labels: start/end/duration
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(8, 18, f"Clip: {sec_to_hms(self.clip_duration)}")
        painter.setFont(QFont("Arial", 9))
        painter.drawText(8, 36, f"Start: {sec_to_hms(self.start_marker)}")
        painter.drawText(8, 52, f"End: {sec_to_hms(self.end_marker)}")
        painter.drawText(w - 150, 18, f"Playhead: {sec_to_hms(self.playhead)}")

    # mouse events to drag markers & place playhead
    def mousePressEvent(self, event):
        x = event.position().x()
        start_x = self.seconds_to_x(self.start_marker)
        end_x = self.seconds_to_x(self.end_marker)
        # small hit radius
        hit_radius = 10
        if abs(x - start_x) <= hit_radius:
            self.dragging = "start"
            self.drag_offset = x - start_x
            return
        if abs(x - end_x) <= hit_radius:
            self.dragging = "end"
            self.drag_offset = x - end_x
            return
        # check if clicking near playhead to start drag playhead or seeking
        ph_x = self.seconds_to_x(self.playhead)
        if abs(x - ph_x) <= hit_radius:
            self.dragging = "playhead"
            self.drag_offset = x - ph_x
            return
        # otherwise treat as seek request (set playhead)
        sec = self.x_to_seconds(x)
        self.playhead = sec
        self.seek_requested.emit(self.playhead)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        x = event.position().x()
        # compute seconds from x, apply drag offset to keep cursor at same relative place
        sec = self.x_to_seconds(x - self.drag_offset)
        sec = max(0.0, min(self.clip_duration, sec))
        if self.dragging == "start":
            # do not surpass end_marker
            self.start_marker = min(sec, self.end_marker)
            self.range_changed.emit(self.start_marker, self.end_marker)
            self.update()
        elif self.dragging == "end":
            self.end_marker = max(sec, self.start_marker)
            self.range_changed.emit(self.start_marker, self.end_marker)
            self.update()
        elif self.dragging == "playhead":
            self.playhead = sec
            self.seek_requested.emit(self.playhead)
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.pixels_per_second = min(self.max_pps, self.pixels_per_second * 1.15)
        else:
            self.pixels_per_second = max(self.min_pps, self.pixels_per_second / 1.15)
        self.update()

# ---------- Main Editor Window ----------

class VideoEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyCut - CapCut-style Video Editor (PyQt6 + MoviePy)")
        self.setGeometry(80, 80, 1280, 780)
        self.setAcceptDrops(True)

        # playback & editing data
        self.clip: Optional[VideoFileClip] = None
        self.clip_path: Optional[str] = None
        self.is_playing = False
        self.preview_fps = 12  # preview FPS (adjust for performance)
        self._last_preview_time = 0.0
        self._play_start_sys = None  # system time when playback started
        self._play_offset = 0.0      # offset at play start (seconds)

        # Build UI
        self._build_ui()

        # QTimer for preview updates; separate from timeline refresh
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / max(1, self.preview_fps)))
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_v = QVBoxLayout()
        central.setLayout(main_v)

        # Top bar + preview row
        top_row = QHBoxLayout()
        main_v.addLayout(top_row)

        # left: file / transport controls (stacked)
        left_column = QVBoxLayout()
        top_row.addLayout(left_column, 1)

        file_row = QHBoxLayout()
        left_column.addLayout(file_row)
        self.import_btn = QPushButton("File → Import")
        self.import_btn.clicked.connect(self.open_file_dialog)
        file_row.addWidget(self.import_btn)
        self.export_btn = QPushButton("Export Trimmed")
        self.export_btn.clicked.connect(self.export_trimmed)
        file_row.addWidget(self.export_btn)

        # playback controls
        transport_row = QHBoxLayout()
        left_column.addLayout(transport_row)
        self.play_btn = QPushButton("Play ▶")
        self.play_btn.clicked.connect(self.play_pause)
        transport_row.addWidget(self.play_btn)
        self.stop_btn = QPushButton("Stop ◼")
        self.stop_btn.clicked.connect(self.stop)
        transport_row.addWidget(self.stop_btn)
        self.step_back_btn = QPushButton("◀ 1s")
        self.step_back_btn.clicked.connect(lambda: self.step_by(-1.0))
        transport_row.addWidget(self.step_back_btn)
        self.step_forward_btn = QPushButton("1s ▶")
        self.step_forward_btn.clicked.connect(lambda: self.step_by(1.0))
        transport_row.addWidget(self.step_forward_btn)

        # current time / duration
        time_row = QHBoxLayout()
        left_column.addLayout(time_row)
        self.current_time_label = QLabel("0:00")
        time_row.addWidget(self.current_time_label)
        self.duration_label = QLabel("/ 0:00")
        time_row.addWidget(self.duration_label)

        left_column.addStretch()

        # center: preview widget (top middle) - big as requested
        center_column = QVBoxLayout()
        top_row.addLayout(center_column, 3)
        self.preview_label = QLabel("Preview (load a video)")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(560, 320)
        self.preview_label.setStyleSheet("background-color: #111; color: #aaa; border: 1px solid #333;")
        center_column.addWidget(self.preview_label)

        # right: markers and zoom controls
        right_column = QVBoxLayout()
        top_row.addLayout(right_column, 1)
        marker_row = QVBoxLayout()
        right_column.addLayout(marker_row)
        self.start_label = QLabel("Start: 0:00")
        marker_row.addWidget(self.start_label)
        self.end_label = QLabel("End: 0:00")
        marker_row.addWidget(self.end_label)

        zoom_row = QHBoxLayout()
        right_column.addLayout(zoom_row)
        zoom_row.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 100)
        self.zoom_slider.setValue(28)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        zoom_row.addWidget(self.zoom_slider)

        right_column.addStretch()

        # timeline widget (big, labeled)
        self.timeline = TimelineWidget()
        main_v.addWidget(self.timeline)

        # bottom: timeline playhead slider + labels
        bottom_row = QHBoxLayout()
        main_v.addLayout(bottom_row)
        self.playhead_slider = QSlider(Qt.Orientation.Horizontal)
        self.playhead_slider.setRange(0, 1000)
        self.playhead_slider.setEnabled(False)
        self.playhead_slider.sliderMoved.connect(self._slider_moved)
        bottom_row.addWidget(QLabel("Seek"))
        bottom_row.addWidget(self.playhead_slider)

        # status bar
        self.status_label = QLabel("Ready")
        main_v.addWidget(self.status_label)

        # connect timeline signals
        self.timeline.range_changed.connect(self._on_range_changed)
        self.timeline.seek_requested.connect(self._seek_to)

    # ---------- File operations and drag/drop ----------

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Video", "", "Video Files (*.mp4 *.mov *.avi *.mkv)", options=QFileDialog.Option(0))
        if file_name:
            self.load_clip(file_name)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                self.load_clip(path)

    def load_clip(self, path: str):
        if VideoFileClip is None:
            QMessageBox.critical(self, "Missing dependency", "MoviePy could not be imported. Install moviepy in your venv.")
            return
        # try to load video -> protect UI from long errors
        self.status_label.setText("Loading video...")
        QApplication.processEvents()
        try:
            clip = VideoFileClip(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Could not load video:\n{e}")
            self.status_label.setText("Ready")
            return

        # assign
        self.clip = clip
        self.clip_path = path
        self.is_playing = False
        self._play_offset = 0.0
        self._play_start_sys = None
        self.timeline.set_clip_duration(self.clip.duration)
        # update labels, slider ranges
        self.playhead_slider.setEnabled(True)
        self.playhead_slider.setRange(0, int(max(1, math.ceil(self.clip.duration * 1000))))
        self.playhead_slider.setValue(0)
        self.duration_label.setText("/ " + sec_to_hms(self.clip.duration))
        self.status_label.setText(f"Loaded: {path.split('/')[-1]} ({self.clip.duration:.2f}s)")

        # update preview first frame
        try:
            f = self.clip.get_frame(0.0)
            pm = frame_to_qpixmap(f, self.preview_label.width(), self.preview_label.height())
            self.preview_label.setPixmap(pm)
        except Exception:
            self.preview_label.setText("Preview unavailable for this format.")

        # sync timeline markers
        self.timeline.playhead = 0.0
        self.timeline.start_marker = 0.0
        self.timeline.end_marker = self.clip.duration
        self.timeline.update()
        self._update_marker_labels()

    # ---------- Playback & preview ----------

    def play_pause(self):
        if not self.clip:
            return
        if self.is_playing:
            # pause
            self._pause_playback()
        else:
            # start playback from timeline.playhead
            self._start_playback_at(self.timeline.playhead)

    def _start_playback_at(self, start_sec: float):
        self.is_playing = True
        self._play_offset = float(start_sec)
        self._play_start_sys = time.time()
        self.play_btn.setText("Pause ❚❚")
        self.status_label.setText("Playing")
        # ensure timer running
        if not self.timer.isActive():
            self.timer.start()

    def _pause_playback(self):
        if not self.is_playing:
            return
        # compute new playhead position
        elapsed = time.time() - (self._play_start_sys or time.time())
        current = self._play_offset + elapsed
        self.timeline.playhead = float(current)
        self.is_playing = False
        self._play_start_sys = None
        self.play_btn.setText("Play ▶")
        self.status_label.setText("Paused")

    def stop(self):
        if not self.clip:
            return
        self.is_playing = False
        self._play_start_sys = None
        self.timeline.playhead = 0.0
        self._play_offset = 0.0
        self.play_btn.setText("Play ▶")
        self._update_ui_for_playhead()

    def step_by(self, seconds: float):
        if not self.clip:
            return
        newt = self.timeline.playhead + seconds
        newt = max(0.0, min(self.clip.duration, newt))
        self._seek_to(newt)

    def _on_timer(self):
        if not self.clip:
            return
        # update playhead if playing
        if self.is_playing and (self._play_start_sys is not None):
            elapsed = time.time() - self._play_start_sys
            current = self._play_offset + elapsed
            # stop if beyond end marker or clip
            if current >= min(self.clip.duration, self.timeline.end_marker):
                self.stop()
                return
            self.timeline.playhead = float(current)
        # update preview but limit fps to preview_fps
        now = time.time()
        if (now - self._last_preview_time) < (1.0 / max(1, self.preview_fps)):
            return
        self._last_preview_time = now
        self._update_preview_frame()

    def _update_preview_frame(self):
        t = float(self.timeline.playhead)
        if not self.clip:
            return
        # bound t
        if t < 0.0:
            t = 0.0
        if t > self.clip.duration:
            t = self.clip.duration
        # fetch frame
        try:
            frame = self.clip.get_frame(t)
            pm = frame_to_qpixmap(frame, self.preview_label.width(), self.preview_label.height())
            self.preview_label.setPixmap(pm)
        except Exception:
            self.preview_label.setText("Preview unavailable (codec may not be supported)")

        # update sliders & labels
        self._update_ui_for_playhead()

    def _update_ui_for_playhead(self):
        self.current_time_label.setText(sec_to_hms(self.timeline.playhead))
        # update timeline widget & playhead slider
        self.timeline.update()
        try:
            slider_val = int(self.timeline.playhead * 1000)
            self.playhead_slider.blockSignals(True)
            self.playhead_slider.setValue(slider_val)
            self.playhead_slider.blockSignals(False)
        except Exception:
            pass

    def _slider_moved(self, value):
        if not self.clip:
            return
        # map slider value (0..ms) to seconds and seek
        sec = float(value) / 1000.0
        self._seek_to(sec)

    def _seek_to(self, sec: float):
        sec = max(0.0, min(self.clip.duration if self.clip else 0.0, sec))
        self.timeline.playhead = sec
        # if playing, restart play timer relative to new offset
        if self.is_playing:
            self._play_offset = sec
            self._play_start_sys = time.time()
        self._update_ui_for_playhead()

    # ---------- Range changes (start/end markers) ----------

    def _on_range_changed(self, start_sec: float, end_sec: float):
        self.start_label.setText(f"Start: {sec_to_hms(start_sec)}")
        self.end_label.setText(f"End: {sec_to_hms(end_sec)}")
        self.timeline.start_marker = start_sec
        self.timeline.end_marker = end_sec
        # if playhead outside new range and playing, clamp
        if self.timeline.playhead < start_sec:
            self._seek_to(start_sec)
        if self.timeline.playhead > end_sec:
            self._seek_to(end_sec)

    def _update_marker_labels(self):
        self.start_label.setText(f"Start: {sec_to_hms(self.timeline.start_marker)}")
        self.end_label.setText(f"End: {sec_to_hms(self.timeline.end_marker)}")

    # ---------- Zoom control ----------

    def _zoom_changed(self, value):
        # map slider value (0..100) to pixels_per_second range
        minv, maxv = 40.0, 400.0
        frac = value / 100.0
        pps = minv + (maxv - minv) * frac
        self.timeline.pixels_per_second = pps
        self.timeline.update()

    # ---------- Export / Cut ----------

    def export_trimmed(self):
        if not self.clip:
            QMessageBox.information(self, "No clip", "Load a clip before exporting.")
            return
        start = float(self.timeline.start_marker)
        end = float(self.timeline.end_marker)
        if end - start <= 0.01:
            QMessageBox.warning(self, "Invalid range", "Select a valid start and end range before exporting.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Trimmed Clip", "", "MP4 Files (*.mp4)", options=QFileDialog.Option(0))
        if not save_path:
            return
        # Ask the user to confirm long operations
        self.status_label.setText("Exporting... (this may take a while)")
        QApplication.processEvents()
        try:
            sub = self.clip.subclip(start, end)
            # use codec and ffmpeg defaults; can be customized
            sub.write_videofile(save_path, threads=4, logger=None)
            self.status_label.setText(f"Exported to {save_path}")
            QMessageBox.information(self, "Export done", f"Trimmed clip exported to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Failed to export: {e}")
            self.status_label.setText("Export failed")

    # ---------- Cutting (in-place trimming) ----------

    def cut_in_place(self):
        if not self.clip:
            return
        start = float(self.timeline.start_marker); end = float(self.timeline.end_marker)
        if end <= start:
            QMessageBox.warning(self, "Invalid range", "End must be after start.")
            return
        try:
            new_clip = self.clip.subclip(start, end)
            # replace current clip
            self.clip.close()
            self.clip = new_clip
            self.clip_path = None
            self.timeline.set_clip_duration(self.clip.duration)
            self.duration_label.setText("/ " + sec_to_hms(self.clip.duration))
            self.status_label.setText("Cut applied to current clip (in-memory)")
            # reset playhead range
            self.timeline.playhead = 0.0
            self._seek_to(0.0)
        except Exception as e:
            QMessageBox.critical(self, "Cut error", f"Cut failed: {e}")

# ---------- Run application ----------

def main():
    app = QApplication(sys.argv)
    w = VideoEditorWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
