# AI Monitor

Real-time NVIDIA GPU monitoring dashboard built with PyQt6 and pyqtgraph.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey)

## Features

- **Live stats** — GPU utilization, memory, temperature, fan speed, power draw, clock speeds
- **History graphs** — 5-minute rolling charts for utilization, temperature, memory, and power
- **Process table** — lists all GPU processes (compute + graphics) with memory usage
- **Fan control** — manual fan speed override via `nvidia-settings` (auto/manual toggle, presets)
- **System tray** — minimize to tray with tooltip showing current stats
- **Adjustable refresh** — 0.5 s to 5 s polling interval

## Requirements

- Linux with NVIDIA GPU
- `nvidia-smi` (included with NVIDIA drivers)
- `nvidia-settings` (optional, for fan control — `sudo apt install nvidia-settings`)
- Python 3.10+

## Install

### From source

```bash
git clone https://github.com/charettep/ai-monitor.git
cd ai-monitor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### As a package

```bash
pip install .
```

This installs an `ai-monitor` command you can run from anywhere.

## Usage

```bash
# From source
python3 ai_monitor.py

# If installed as a package
ai-monitor
```

The window opens showing live GPU data. Close the window to minimize to the system tray — right-click the tray icon to quit.

### Fan control

The fan control panel on the right requires `nvidia-settings` and an X11 display. Click **Manual**, adjust the slider or pick a preset, then **Apply Speed**. Click **Auto** to restore automatic fan control.

## Standalone binary

Pre-built Linux binaries are available on the [Releases](https://github.com/charettep/ai-monitor/releases) page — no Python installation needed.

To build locally:

```bash
pip install pyinstaller
pyinstaller ai-monitor.spec
# Output: dist/ai-monitor
```

## Architecture

Single-file app (`ai_monitor.py`) with these layers:

| Layer | Description |
|---|---|
| `run_cmd()` | Safe subprocess wrapper (no `shell=True`) |
| `parse_gpu_stats()` / `parse_processes()` | Parse `nvidia-smi` CSV output into dicts |
| `fan_set_manual()` / `fan_set_auto()` | Fan control via `nvidia-settings` |
| `GpuPoller` / `ProcessPoller` | QThread workers emitting signals on intervals |
| `StatBar` / `make_graph()` | Custom widgets (progress bars, pyqtgraph charts) |
| `FanControlPanel` | Fan speed UI with slider, presets, mode toggle |
| `MainWindow` | Assembles everything — header, stats, graphs, process table, footer |

## License

[MIT](LICENSE)
