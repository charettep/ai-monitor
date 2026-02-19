#!/usr/bin/env python3
"""ai-monitor — GPU Dashboard with PyQt6 + pyqtgraph live monitoring and fan control."""

import subprocess
import sys
import os
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import (
    QThread, Qt, pyqtSignal
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QFont, QAction
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSlider, QGroupBox, QSplitter, QSystemTrayIcon,
    QMenu, QPushButton, QFrame, QSizePolicy
)

# ─── Color palette ───────────────────────────────────────────────────────────
BG      = "#1a1a2e"
CARD    = "#16213e"
CARD2   = "#0f3460"
ACCENT  = "#76b900"   # NVIDIA green
WARNING = "#e94560"   # red-orange
TEXT    = "#e0e0e0"
SUBTEXT = "#8892a4"
BORDER  = "#2a2a4a"

HISTORY_LEN = 300  # 5 min at 1 s intervals

# ─── Safe subprocess helper (no shell=True) ───────────────────────────────────

def run_cmd(args: list[str]) -> str:
    """Run a command with argument list (no shell) — immune to injection."""
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.stdout.decode(errors="replace").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return ""


# ─── nvidia-smi helpers ───────────────────────────────────────────────────────

GPU_QUERY_FIELDS = (
    "name,driver_version,"
    "temperature.gpu,fan.speed,"
    "power.draw,power.limit,"
    "memory.used,memory.total,"
    "utilization.gpu,utilization.memory,"
    "clocks.current.graphics,clocks.current.memory"
)


def _get_cuda_version() -> str:
    """Parse CUDA version from 'nvidia-smi --query' (not available in --query-gpu)."""
    raw = run_cmd(["nvidia-smi", "--query"])
    for line in raw.splitlines():
        if "CUDA Version" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                return parts[-1].strip()
    return "N/A"


def safe_float(val: str, default: float = 0.0) -> float:
    """Convert nvidia-smi field to float, returning default for N/A values."""
    v = val.strip()
    if not v or v in ("[N/A]", "N/A", "[Not Supported]", "Not Supported", "[Unknown Error]"):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def parse_gpu_stats() -> dict | None:
    raw = run_cmd([
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 12:
        return None
    return {
        "name":        parts[0],
        "driver":      parts[1],
        "temp":        safe_float(parts[2]),
        "fan":         safe_float(parts[3]),
        "power_draw":  safe_float(parts[4]),
        "power_limit": safe_float(parts[5], default=200.0),
        "mem_used":    safe_float(parts[6]),
        "mem_total":   safe_float(parts[7], default=1.0),
        "gpu_util":    safe_float(parts[8]),
        "mem_util":    safe_float(parts[9]),
        "clk_gpu":     safe_float(parts[10]),
        "clk_mem":     safe_float(parts[11]),
    }


def parse_processes() -> list[dict]:
    procs: list[dict] = []

    # Compute apps
    raw_compute = run_cmd([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    for line in raw_compute.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                procs.append({
                    "pid":    int(parts[0]),
                    "name":   os.path.basename(parts[1]),
                    "type":   "Compute",
                    "mem_mb": int(parts[2]) if parts[2] not in ("[N/A]", "N/A") else 0,
                })
            except ValueError:
                pass

    # Graphics apps via pmon
    raw_pmon = run_cmd(["nvidia-smi", "pmon", "-c", "1", "-s", "m"])
    seen_pids = {p["pid"] for p in procs}
    for line in raw_pmon.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        if len(cols) < 4:
            continue
        try:
            pid = int(cols[1])
        except ValueError:
            continue
        if pid <= 0 or pid in seen_pids:
            continue
        fb_mem = 0
        try:
            fb_mem = int(cols[3]) if cols[3] != "-" else 0
        except ValueError:
            pass
        # Safe ps lookup — pid is an int, converted to str, no injection risk
        name_raw = run_cmd(["ps", "-p", str(pid), "-o", "comm="])
        procs.append({
            "pid":    pid,
            "name":   os.path.basename(name_raw) if name_raw else "unknown",
            "type":   "Graphics",
            "mem_mb": fb_mem,
        })
        seen_pids.add(pid)

    return procs


# ─── Fan control via nvidia-settings ─────────────────────────────────────────

def _nvidia_settings(*args: str) -> str:
    display = os.environ.get("DISPLAY", ":0")
    return run_cmd(["nvidia-settings", f"--display={display}", *args])


def fan_set_manual(speed_pct: int, gpu_index: int = 0) -> bool:
    """Enable manual fan control and set target speed (0–100%)."""
    speed_pct = max(0, min(100, speed_pct))
    r1 = _nvidia_settings(f"-a", f"[gpu:{gpu_index}]/GPUFanControlState=1")
    r2 = _nvidia_settings(f"-a", f"[fan:{gpu_index}]/GPUTargetFanSpeed={speed_pct}")
    return "assigned" in r1.lower() or "assigned" in r2.lower()


def fan_set_auto(gpu_index: int = 0) -> bool:
    """Restore automatic fan control."""
    r = _nvidia_settings("-a", f"[gpu:{gpu_index}]/GPUFanControlState=0")
    return "assigned" in r.lower()


def nvidia_settings_available() -> bool:
    return bool(run_cmd(["which", "nvidia-settings"]))


# ─── Worker threads ───────────────────────────────────────────────────────────

class GpuPoller(QThread):
    data_ready = pyqtSignal(dict)

    def __init__(self, interval_ms: int = 1000):
        super().__init__()
        self._interval = interval_ms
        self._running = True

    def set_interval(self, ms: int):
        self._interval = ms

    def run(self):
        while self._running:
            stats = parse_gpu_stats()
            if stats:
                self.data_ready.emit(stats)
            self.msleep(self._interval)

    def stop(self):
        self._running = False
        self.wait()


class ProcessPoller(QThread):
    data_ready = pyqtSignal(list)

    def __init__(self, interval_ms: int = 3000):
        super().__init__()
        self._interval = interval_ms
        self._running = True

    def run(self):
        while self._running:
            self.data_ready.emit(parse_processes())
            self.msleep(self._interval)

    def stop(self):
        self._running = False
        self.wait()


# ─── Custom widgets ───────────────────────────────────────────────────────────

class StatBar(QWidget):
    """Labeled progress bar with value text overlay."""

    def __init__(self, label: str, unit: str = "", color: str = ACCENT, parent=None):
        super().__init__(parent)
        self._unit = unit
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:12px;")
        layout.addWidget(lbl)

        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(18)
        self._bar_color = color
        self._apply_bar_style(color)
        layout.addWidget(self._bar, 1)

        self._val_lbl = QLabel("—")
        self._val_lbl.setFixedWidth(150)
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val_lbl.setStyleSheet(f"color:{TEXT}; font-size:12px; font-family:monospace;")
        layout.addWidget(self._val_lbl)

    def _apply_bar_style(self, color: str):
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background: {CARD2};
                border: 1px solid {BORDER};
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 2px;
            }}
        """)

    def update(self, value: float, maximum: float, text: str = ""):
        pct = int(value / maximum * 100) if maximum else 0
        self._bar.setMaximum(100)
        self._bar.setValue(pct)
        color = WARNING if pct >= 85 else self._bar_color
        self._apply_bar_style(color)
        self._val_lbl.setText(text if text else f"{value:.0f}{self._unit}")


def make_graph(title: str, color: str, y_label: str, y_range: tuple = (0, 100)):
    """Return (PlotWidget, curve, baseline) pre-configured for dark theme.

    Both curve and baseline are persistent PlotDataItems added to the widget.
    FillBetweenItem auto-updates when either item's data changes via setData().
    """
    pg.setConfigOptions(antialias=True)
    pw = pg.PlotWidget()
    pw.setBackground(CARD)
    pw.setTitle(title, color=TEXT, size="10pt")
    pw.showGrid(x=False, y=True, alpha=0.15)
    pw.setYRange(*y_range, padding=0.05)
    pw.setXRange(-HISTORY_LEN, 0, padding=0.02)

    left = pw.getAxis("left")
    left.setLabel(y_label, color=SUBTEXT)
    left.setTextPen(pg.mkPen(SUBTEXT))

    bottom = pw.getAxis("bottom")
    bottom.setLabel("seconds ago", color=SUBTEXT)
    bottom.setTextPen(pg.mkPen(SUBTEXT))
    bottom.enableAutoSIPrefix(False)   # prevent "x0.001" auto-scaling

    pw.getPlotItem().layout.setContentsMargins(4, 4, 4, 4)

    # Persistent baseline (invisible line at y=0) — added to plot so
    # FillBetweenItem receives sigPlotChanged updates automatically
    baseline = pw.plot(pen=None)

    fill = pg.FillBetweenItem(
        # curve must be added after fill registration so it renders on top
        baseline,
        baseline,  # placeholder; overwritten below
        brush=pg.mkBrush(QColor(color).darker(300).name() + "80"),
    )
    pw.addItem(fill)

    curve = pw.plot(pen=pg.mkPen(color=color, width=2))
    fill.setCurves(baseline, curve)

    return pw, curve, baseline


# ─── Fan control panel ────────────────────────────────────────────────────────

class FanControlPanel(QGroupBox):
    """Panel for manual GPU fan speed control via nvidia-settings."""

    def __init__(self, parent=None):
        super().__init__("Fan Control", parent)
        self._manual_mode = False
        self._ns_available = nvidia_settings_available()
        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        self.setStyleSheet(f"""
            QGroupBox {{
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                margin-top: 10px;
                font-size: 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                padding: 0 6px;
                color: {ACCENT};
            }}
        """)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._status = QLabel()
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        if not self._ns_available:
            self._status.setText(
                "nvidia-settings not found.\n"
                "Install: sudo apt install nvidia-settings"
            )
            self._status.setStyleSheet(f"color:{WARNING}; font-size:11px;")
            return

        self._status.setStyleSheet(f"color:{SUBTEXT}; font-size:11px;")

        # Current fan readout
        self._cur_lbl = QLabel("Current: — %")
        self._cur_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cur_lbl.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:bold;")
        layout.addWidget(self._cur_lbl)

        # Mode toggle row
        self._auto_btn   = QPushButton("Auto")
        self._manual_btn = QPushButton("Manual")
        for btn in (self._auto_btn, self._manual_btn):
            btn.setFixedHeight(28)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._auto_btn)
        mode_row.addWidget(self._manual_btn)
        layout.addLayout(mode_row)

        # Target speed slider
        self._slider_lbl = QLabel("Target: 50%")
        self._slider_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slider_lbl.setStyleSheet(f"color:{TEXT}; font-size:12px;")
        layout.addWidget(self._slider_lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.setTickInterval(10)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setEnabled(False)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {CARD2}; height: 6px; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 16px; height: 16px;
                margin: -5px 0; border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT}; border-radius: 3px;
            }}
            QSlider::disabled {{ opacity: 0.4; }}
        """)
        layout.addWidget(self._slider)

        # Apply
        self._apply_btn = QPushButton("Apply Speed")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setFixedHeight(30)
        layout.addWidget(self._apply_btn)

        # Presets
        preset_row = QHBoxLayout()
        for label, val in [("0%", 0), ("30%", 30), ("50%", 50), ("75%", 75), ("100%", 100)]:
            pb = QPushButton(label)
            pb.setFixedHeight(24)
            pb.clicked.connect(lambda _, v=val: self._set_preset(v))
            preset_row.addWidget(pb)
        layout.addLayout(preset_row)

        self._result_lbl = QLabel("")
        self._result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_lbl.setStyleSheet(f"color:{ACCENT}; font-size:11px;")
        self._result_lbl.setWordWrap(True)
        layout.addWidget(self._result_lbl)

        layout.addStretch()

        # Connect
        self._auto_btn.clicked.connect(self._enable_auto)
        self._manual_btn.clicked.connect(self._enable_manual)
        self._slider.valueChanged.connect(
            lambda v: self._slider_lbl.setText(f"Target: {v}%")
        )
        self._apply_btn.clicked.connect(self._apply_speed)

        self._refresh_mode_ui()

    def _btn_style(self, bg: str) -> str:
        return (
            f"QPushButton {{ background:{bg}; color:{TEXT}; border:1px solid {BORDER}; "
            f"border-radius:4px; font-size:11px; padding:3px 8px; }}"
            f"QPushButton:hover {{ border-color:{ACCENT}; }}"
            f"QPushButton:disabled {{ color:{SUBTEXT}; }}"
        )

    def _set_preset(self, val: int):
        self._slider.setValue(val)
        if self._manual_mode:
            self._apply_speed()

    def _enable_auto(self):
        ok = fan_set_auto()
        self._manual_mode = False
        self._refresh_mode_ui()
        self._show_result("Switched to Auto" if ok else "Failed — check nvidia-settings", ok)

    def _enable_manual(self):
        self._manual_mode = True
        self._refresh_mode_ui()
        self._show_result("Manual mode active. Set speed and Apply.", True)

    def _apply_speed(self):
        speed = self._slider.value()
        ok = fan_set_manual(speed)
        self._show_result(
            f"Set to {speed}%" if ok else "Failed — ensure DISPLAY=:0 and X is running",
            ok,
        )

    def _show_result(self, msg: str, success: bool):
        color = ACCENT if success else WARNING
        self._result_lbl.setStyleSheet(f"color:{color}; font-size:11px;")
        self._result_lbl.setText(msg)

    def _refresh_mode_ui(self):
        if not self._ns_available:
            return
        is_manual = self._manual_mode
        self._slider.setEnabled(is_manual)
        self._apply_btn.setEnabled(is_manual)
        self._auto_btn.setStyleSheet(self._btn_style(ACCENT if not is_manual else CARD2))
        self._manual_btn.setStyleSheet(self._btn_style(ACCENT if is_manual else CARD2))

    def set_current_fan(self, pct: float):
        if hasattr(self, "_cur_lbl"):
            self._cur_lbl.setText(f"Current: {pct:.0f}%")


# ─── Main window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Monitor")
        self.resize(1100, 780)
        self.setMinimumSize(900, 650)

        self._history = {
            k: deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
            for k in ("gpu_util", "temp", "mem_used", "power_draw")
        }
        self._x = np.arange(-HISTORY_LEN + 1, 1, dtype=float)

        self._cuda_ver = _get_cuda_version()  # fetch once at startup

        self._build_ui()
        self._apply_stylesheet()
        self._build_tray()

        # Pollers
        self._gpu_poller = GpuPoller(1000)
        self._gpu_poller.data_ready.connect(self._on_gpu_data)
        self._gpu_poller.start()

        self._proc_poller = ProcessPoller(3000)
        self._proc_poller.data_ready.connect(self._on_proc_data)
        self._proc_poller.start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _card(self, title: str = "") -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet(f"""
            QGroupBox {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 6px;
                margin-top: 10px;
                font-size: 12px;
                font-weight: bold;
                color: {ACCENT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                padding: 0 6px;
            }}
        """)
        return box

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        top_half = QWidget()
        top_layout = QHBoxLayout(top_half)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        top_layout.addWidget(self._build_stats_panel(), 1)
        top_layout.addWidget(self._build_fan_control_panel(), 0)

        splitter.addWidget(top_half)
        splitter.addWidget(self._build_graphs_panel())
        splitter.addWidget(self._build_process_panel())
        splitter.setSizes([240, 320, 200])

        root.addWidget(splitter, 1)
        root.addWidget(self._build_footer())

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(
            f"background:{CARD}; border:1px solid {BORDER}; border-radius:6px;"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 0, 12, 0)

        self._gpu_name_lbl = QLabel("Detecting GPU…")
        self._gpu_name_lbl.setStyleSheet(
            f"color:{ACCENT}; font-size:14px; font-weight:bold;"
        )
        hl.addWidget(self._gpu_name_lbl)
        hl.addStretch()

        self._driver_lbl = QLabel()
        self._driver_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:11px;")
        hl.addWidget(self._driver_lbl)
        return header

    def _build_stats_panel(self) -> QGroupBox:
        box = self._card("Live Stats")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._bar_gpu  = StatBar("GPU Util",  "%",    ACCENT)
        self._bar_mem  = StatBar("Memory",    " MiB", "#4fc3f7")
        self._bar_temp = StatBar("Temp",      "°C",   "#ff7043")
        self._bar_fan  = StatBar("Fan",       "%",    "#ab47bc")
        self._bar_pwr  = StatBar("Power",     "W",    "#ffd54f")

        for bar in (self._bar_gpu, self._bar_mem, self._bar_temp,
                    self._bar_fan, self._bar_pwr):
            layout.addWidget(bar)

        layout.addStretch()

        self._clk_lbl = QLabel("GPU: — MHz  |  Mem: — MHz")
        self._clk_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; padding:4px 0;")
        layout.addWidget(self._clk_lbl)
        return box

    def _build_fan_control_panel(self) -> FanControlPanel:
        self._fan_panel = FanControlPanel()
        self._fan_panel.setFixedWidth(240)
        return self._fan_panel

    def _build_graphs_panel(self) -> QGroupBox:
        box = self._card("History (5 min)")
        grid = QGridLayout(box)
        grid.setSpacing(4)

        self._gw_util, self._curve_util, self._base_util = make_graph(
            "GPU Utilization", ACCENT, "%", (0, 100))
        self._gw_temp, self._curve_temp, self._base_temp = make_graph(
            "Temperature", "#ff7043", "°C", (20, 90))
        self._gw_mem,  self._curve_mem,  self._base_mem  = make_graph(
            "Memory Used", "#4fc3f7", "MiB", (0, 1))
        self._gw_pwr,  self._curve_pwr,  self._base_pwr  = make_graph(
            "Power Draw", "#ffd54f", "W", (0, 200))

        grid.addWidget(self._gw_util, 0, 0)
        grid.addWidget(self._gw_temp, 0, 1)
        grid.addWidget(self._gw_mem,  1, 0)
        grid.addWidget(self._gw_pwr,  1, 1)
        return box

    def _build_process_panel(self) -> QGroupBox:
        box = self._card("GPU Processes")
        layout = QVBoxLayout(box)

        self._proc_table = QTableWidget(0, 4)
        self._proc_table.setHorizontalHeaderLabels(
            ["PID", "Name", "Type", "GPU Mem (MiB)"]
        )
        self._proc_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._proc_table.horizontalHeader().setDefaultSectionSize(110)
        self._proc_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._proc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._proc_table.setSortingEnabled(True)
        self._proc_table.verticalHeader().setVisible(False)
        self._proc_table.setAlternatingRowColors(True)
        self._proc_table.setStyleSheet(f"""
            QTableWidget {{
                background: {CARD}; alternate-background-color: {CARD2};
                color: {TEXT}; gridline-color: {BORDER};
                border: none; font-size: 12px;
            }}
            QHeaderView::section {{
                background: {CARD2}; color: {ACCENT};
                border: 1px solid {BORDER}; padding: 4px; font-weight: bold;
            }}
            QTableWidget::item:selected {{ background: {CARD2}; color: {ACCENT}; }}
        """)
        layout.addWidget(self._proc_table)
        return box

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setFixedHeight(36)
        footer.setStyleSheet(
            f"background:{CARD}; border:1px solid {BORDER}; border-radius:6px;"
        )
        hl = QHBoxLayout(footer)
        hl.setContentsMargins(12, 0, 12, 0)

        lbl = QLabel("Refresh interval:")
        lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:11px;")
        hl.addWidget(lbl)

        self._interval_slider = QSlider(Qt.Orientation.Horizontal)
        self._interval_slider.setRange(5, 50)   # ×100 ms → 0.5–5 s
        self._interval_slider.setValue(10)
        self._interval_slider.setFixedWidth(140)
        self._interval_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background:{CARD2}; height:4px; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{ACCENT}; width:12px; height:12px;
                                          margin:-4px 0; border-radius:6px; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
        """)
        self._interval_slider.valueChanged.connect(self._on_interval_changed)
        hl.addWidget(self._interval_slider)

        self._interval_lbl = QLabel("1.0 s")
        self._interval_lbl.setStyleSheet(f"color:{TEXT}; font-size:11px; min-width:40px;")
        hl.addWidget(self._interval_lbl)

        hl.addStretch()

        self._status_lbl = QLabel("Starting…")
        self._status_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:10px;")
        hl.addWidget(self._status_lbl)
        return footer

    # ── Global stylesheet ──────────────────────────────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
            QSplitter::handle {{ background: {BORDER}; }}
            QLabel {{ color: {TEXT}; }}
            QScrollBar:vertical {{
                background: {CARD}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER}; border-radius: 4px; min-height: 20px;
            }}
        """)

    # ── System tray ───────────────────────────────────────────────────────────

    def _build_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return

        pix = QPixmap(16, 16)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setBrush(QColor(ACCENT))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, 16, 16, 3, 3)
        painter.setPen(QColor("#000"))
        painter.setFont(QFont("monospace", 6, QFont.Weight.Bold))
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "GPU")
        painter.end()

        self._tray = QSystemTrayIcon(QIcon(pix), self)
        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{ background:{CARD}; color:{TEXT}; border:1px solid {BORDER}; }}
            QMenu::item:selected {{ background:{CARD2}; }}
        """)
        show_act = QAction("Show / Hide", self)
        show_act.triggered.connect(self._toggle_visibility)
        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(QApplication.instance().quit)
        menu.addAction(show_act)
        menu.addSeparator()
        menu.addAction(quit_act)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._toggle_visibility()

    # ── Data slots ────────────────────────────────────────────────────────────

    def _on_gpu_data(self, d: dict):
        # Header
        self._gpu_name_lbl.setText(d["name"])
        self._driver_lbl.setText(
            f"Driver {d['driver']}  |  CUDA {self._cuda_ver}"
        )

        # Stat bars
        self._bar_gpu.update(d["gpu_util"],  100,
                             f"{d['gpu_util']:.0f}%")
        self._bar_mem.update(d["mem_used"], d["mem_total"],
                             f"{d['mem_used']:.0f} / {d['mem_total']:.0f} MiB")
        self._bar_temp.update(d["temp"],    100,
                              f"{d['temp']:.0f}°C")
        self._bar_fan.update(d["fan"],      100,
                             f"{d['fan']:.0f}%")
        self._bar_pwr.update(d["power_draw"], d["power_limit"],
                             f"{d['power_draw']:.1f} / {d['power_limit']:.0f} W")

        self._clk_lbl.setText(
            f"GPU: {d['clk_gpu']:.0f} MHz  |  Mem: {d['clk_mem']:.0f} MHz"
        )
        self._fan_panel.set_current_fan(d["fan"])

        # Append to rolling history
        for key in ("gpu_util", "temp", "mem_used", "power_draw"):
            self._history[key].append(d[key])

        x     = self._x
        y_util = np.array(self._history["gpu_util"])
        y_temp = np.array(self._history["temp"])
        y_mem  = np.array(self._history["mem_used"])
        y_pwr  = np.array(self._history["power_draw"])
        zeros  = np.zeros(HISTORY_LEN)

        # Update curves — baselines use setData() so FillBetweenItem auto-refreshes
        self._curve_util.setData(x, y_util)
        self._base_util.setData(x, zeros)

        self._curve_temp.setData(x, y_temp)
        self._base_temp.setData(x, np.full(HISTORY_LEN, 20.0))

        self._gw_mem.setYRange(0, d["mem_total"] * 1.05, padding=0)
        self._curve_mem.setData(x, y_mem)
        self._base_mem.setData(x, zeros)

        self._gw_pwr.setYRange(0, max(d["power_limit"] * 1.1, 10), padding=0)
        self._curve_pwr.setData(x, y_pwr)
        self._base_pwr.setData(x, zeros)

        # Tray tooltip
        if self._tray:
            self._tray.setToolTip(
                f"{d['name']}\n"
                f"GPU {d['gpu_util']:.0f}%  Temp {d['temp']:.0f}°C\n"
                f"Mem {d['mem_used']:.0f}/{d['mem_total']:.0f} MiB  "
                f"Power {d['power_draw']:.0f}W"
            )

        self._status_lbl.setText(
            f"GPU {d['gpu_util']:.0f}%  "
            f"Temp {d['temp']:.0f}°C  "
            f"Mem {d['mem_used']:.0f} MiB  "
            f"Power {d['power_draw']:.1f}W"
        )

    def _on_proc_data(self, procs: list):
        self._proc_table.setSortingEnabled(False)
        self._proc_table.setRowCount(len(procs))
        for row, p in enumerate(procs):
            items = [
                QTableWidgetItem(str(p["pid"])),
                QTableWidgetItem(p["name"]),
                QTableWidgetItem(p["type"]),
                QTableWidgetItem(str(p["mem_mb"])),
            ]
            for col in (0, 3):
                items[col].setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            type_color = "#4fc3f7" if p["type"] == "Compute" else "#ab47bc"
            items[2].setForeground(QColor(type_color))
            for col, item in enumerate(items):
                self._proc_table.setItem(row, col, item)
        self._proc_table.setSortingEnabled(True)

    # ── Refresh interval ──────────────────────────────────────────────────────

    def _on_interval_changed(self, val: int):
        ms = val * 100
        self._interval_lbl.setText(f"{ms / 1000:.1f} s")
        self._gpu_poller.set_interval(ms)

    # ── Window close → minimize to tray ──────────────────────────────────────

    def closeEvent(self, event):
        if self._tray and self._tray.isVisible():
            self.hide()
            self._tray.showMessage(
                "AI Monitor",
                "Running in the system tray. Right-click the icon to quit.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            event.ignore()
        else:
            self._gpu_poller.stop()
            self._proc_poller.stop()
            event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    app = QApplication(sys.argv)
    app.setApplicationName("AI Monitor")
    app.setQuitOnLastWindowClosed(False)  # keep alive in tray

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
