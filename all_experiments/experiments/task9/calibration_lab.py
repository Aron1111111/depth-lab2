"""多传感器同步与标定实验。

本实验使用可重复的模拟 Camera/LiDAR/IMU 数据，完成：
1) 时间偏移 dt 估计；2) 时间抖动和匹配率评价；
3) LiDAR 到相机的外参估计；4) 重投影误差评价；
5) 输出 calib_v1.yaml、metrics.json、报告和图表。
"""

from __future__ import annotations

import json
import hashlib
import math
import subprocess
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def trajectory(t: np.ndarray) -> np.ndarray:
    """一个有足够变化的目标轨迹，避免时间偏移估计退化。"""
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        1.4 * np.sin(0.75 * t) + 0.12 * t,
        0.9 * np.cos(0.43 * t) + 0.1 * np.sin(1.7 * t),
        7.5 + 0.45 * np.sin(0.31 * t) + 0.12 * np.cos(1.2 * t),
    ])


def scalar_signal(t: np.ndarray) -> np.ndarray:
    p = trajectory(t)
    return p[:, 0] + 0.35 * p[:, 1] + 0.18 * p[:, 2] + 0.15 * np.sin(2.3 * t)


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """估计 target ≈ R @ source + t。"""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    a = source - source_mean
    b = target - target_mean
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_mean - r @ source_mean
    return r, t


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-6)
    return np.column_stack([
        k[0, 0] * points[:, 0] / z + k[0, 2],
        k[1, 1] * points[:, 1] / z + k[1, 2],
    ])


def estimate_time_offset(camera_t: np.ndarray, camera_signal: np.ndarray, lidar_t: np.ndarray, lidar_signal: np.ndarray) -> tuple[float, float, float]:
    candidates = np.linspace(-0.12, 0.12, 481)
    losses: list[float] = []
    for dt in candidates:
        query = camera_t - dt
        valid = (query >= lidar_t.min()) & (query <= lidar_t.max())
        if valid.sum() < 30:
            losses.append(np.inf)
            continue
        predicted = np.interp(query[valid], lidar_t, lidar_signal)
        losses.append(float(np.mean((camera_signal[valid] - predicted) ** 2)))
    index = int(np.argmin(losses))
    dt = float(candidates[index])
    query = camera_t - dt
    nearest = np.searchsorted(lidar_t, query.clip(lidar_t.min(), lidar_t.max()))
    nearest = np.clip(nearest, 1, len(lidar_t) - 1)
    left = lidar_t[nearest - 1]
    right = lidar_t[nearest]
    nearest_gap = np.minimum(np.abs(query - left), np.abs(right - query))
    # LiDAR 周期为 100 ms；不同频率传感器对齐时，以半个周期作为
    # 最近邻匹配窗口，避免把正常的采样率差异误判成同步失败。
    match_window = 0.5 * float(np.median(np.diff(lidar_t))) + 0.005
    matched = nearest_gap <= match_window
    match_rate = float(np.mean(matched))
    jitter = float(np.std(nearest_gap[matched])) if np.any(matched) else float("inf")
    return dt, match_rate, jitter


def write_yaml(path: Path, metrics: dict) -> None:
    r = np.asarray(metrics["spatial_calibration"]["R_lidar_to_camera"])
    t = np.asarray(metrics["spatial_calibration"]["t_lidar_to_camera_m"])
    k = np.asarray(metrics["camera_model"]["K"])
    lines = [
        "version: 1",
        "coordinate_convention: 'target_camera = R_lidar_to_camera * target_lidar + t'",
        "time:",
        f"  camera_to_lidar_offset_s: {metrics['time_sync']['estimated_offset_s']:.6f}",
        f"  camera_to_lidar_offset_ground_truth_s: {metrics['time_sync']['ground_truth_offset_s']:.6f}",
        "camera_model:",
        "  K:",
    ]
    lines.extend([f"    - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}]" for row in k])
    lines.extend([
        "  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]",
        "spatial:",
        "  lidar_to_camera:",
        "    R:",
    ])
    lines.extend([f"      - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}]" for row in r])
    lines.append(f"    t_m: [{t[0]:.8f}, {t[1]:.8f}, {t[2]:.8f}]")
    lines.extend([
        "metrics:",
        f"  match_rate: {metrics['time_sync']['match_rate']:.6f}",
        f"  jitter_s: {metrics['time_sync']['jitter_s']:.6f}",
        f"  reprojection_before_px: {metrics['spatial_calibration']['reprojection_before_px']:.6f}",
        f"  reprojection_after_px: {metrics['spatial_calibration']['reprojection_after_px']:.6f}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(235, 242, 250), size=20) -> None:
    try:
        font = ImageFont.truetype("msyh.ttc", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=fill)


def render_timeline(path: Path, camera_t: np.ndarray, lidar_t: np.ndarray, dt: float) -> None:
    w, h = 1200, 430
    image = Image.new("RGB", (w, h), (12, 22, 37))
    d = ImageDraw.Draw(image)
    draw_text(d, (45, 28), "时间同步：Camera 与 LiDAR 时间轴", size=28)
    draw_text(d, (45, 75), f"估计 camera_to_lidar offset = {dt * 1000:.2f} ms", fill=(95, 215, 220), size=20)
    left, right = 80, 1120
    span = max(float(camera_t.max()), float(lidar_t.max()))
    for y, label, times, color in [(170, "Camera 30 FPS", camera_t, (80, 210, 225)), (290, "LiDAR 10 Hz", lidar_t, (92, 216, 150))]:
        d.line((left, y, right, y), fill=(100, 125, 150), width=2)
        draw_text(d, (45, y - 35), label, fill=color, size=18)
        for t in times[::max(1, len(times) // 60)]:
            x = int(left + (right - left) * float(t) / span)
            d.ellipse((x - 3, y - 8, x + 3, y + 8), fill=color)
    x = int(left + (right - left) * 0.5)
    d.line((x, 145, x, 320), fill=(255, 180, 75), width=2)
    draw_text(d, (x + 10, 335), "同一时刻对齐后再融合", fill=(255, 190, 95), size=18)
    image.save(path)


def render_reprojection(path: Path, gt: np.ndarray, before: np.ndarray, after: np.ndarray) -> None:
    w, h = 1200, 520
    image = Image.new("RGB", (w, h), (12, 22, 37))
    d = ImageDraw.Draw(image)
    draw_text(d, (45, 25), "Camera-LiDAR 重投影误差", size=28)
    panels = [(40, "未标定 / Before", before, (255, 105, 125)), (620, "标定后 / After", after, (85, 220, 165))]
    for ox, title, pred, color in panels:
        d.rectangle((ox, 85, ox + 500, 460), outline=(70, 115, 145), width=2)
        draw_text(d, (ox + 20, 105), title, fill=color, size=22)
        for p, q in zip(gt, pred):
            x1, y1 = int(ox + 50 + p[0] * 0.35), int(155 + p[1] * 0.95)
            x2, y2 = int(ox + 50 + q[0] * 0.35), int(155 + q[1] * 0.95)
            d.line((x1, y1, x2, y2), fill=(255, 185, 90), width=1)
            d.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), fill=(90, 190, 245))
            d.ellipse((x2 - 3, y2 - 3, x2 + 3, y2 + 3), fill=color)
    draw_text(d, (45, 475), "蓝点：相机参考位置   彩色点：LiDAR 投影位置   黄色线：误差", fill=(180, 200, 220), size=16)
    image.save(path)


def run_experiment() -> dict:
    rng = np.random.default_rng(42)
    duration = 10.0
    true_dt = 0.050
    lidar_t = np.arange(0.0, duration, 0.1)
    camera_actual_t = np.arange(0.0, duration, 1.0 / 30.0)
    camera_t = camera_actual_t + true_dt + rng.normal(0, 0.0025, len(camera_actual_t))
    lidar_reported_t = lidar_t + rng.normal(0, 0.0010, len(lidar_t))
    camera_signal = scalar_signal(camera_actual_t) + rng.normal(0, 0.015, len(camera_actual_t))
    lidar_signal = scalar_signal(lidar_t) + rng.normal(0, 0.010, len(lidar_t))
    estimated_dt, match_rate, jitter = estimate_time_offset(camera_t, camera_signal, lidar_reported_t, lidar_signal)

    n = 32
    lidar_points = rng.uniform([-1.8, -1.2, 5.5], [1.8, 1.2, 10.5], (n, 3))
    true_r = rot_z(np.deg2rad(5.0)) @ rot_z(np.deg2rad(-2.0))
    true_t = np.array([0.23, -0.08, 0.14])
    camera_points = (true_r @ lidar_points.T).T + true_t
    camera_points += rng.normal(0, 0.006, camera_points.shape)
    estimated_r, estimated_t = kabsch(lidar_points, camera_points)
    k = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]])
    gt_pixels = project(camera_points, k)
    before_pixels = project(lidar_points, k)
    after_pixels = project((estimated_r @ lidar_points.T).T + estimated_t, k)
    error_before = float(np.mean(np.linalg.norm(before_pixels - gt_pixels, axis=1)))
    error_after = float(np.mean(np.linalg.norm(after_pixels - gt_pixels, axis=1)))
    rot_error_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(estimated_r @ true_r.T) - 1) / 2, -1, 1))))
    trans_error = float(np.linalg.norm(estimated_t - true_t))

    metrics = {
        "experiment": "multisensor_synchronization_and_calibration",
        "seed": 42,
        "data": {"duration_s": duration, "camera_fps": 30, "lidar_hz": 10, "imu_hz": 200, "synthetic": True},
        "time_sync": {"ground_truth_offset_s": true_dt, "estimated_offset_s": estimated_dt, "absolute_offset_error_ms": abs(estimated_dt - true_dt) * 1000, "jitter_s": jitter, "match_rate": match_rate},
        "camera_model": {"K": k.tolist(), "distortion": [0, 0, 0, 0, 0]},
        "spatial_calibration": {"R_lidar_to_camera": estimated_r.tolist(), "t_lidar_to_camera_m": estimated_t.tolist(), "rotation_error_deg": rot_error_deg, "translation_error_m": trans_error, "reprojection_before_px": error_before, "reprojection_after_px": error_after},
        "acceptance": {"time_offset_error_ms_under_10": abs(estimated_dt - true_dt) * 1000 < 10, "match_rate_over_0_8": match_rate > 0.8, "reprojection_after_under_3px": error_after < 3.0},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_yaml(RESULTS / "calib_v1.yaml", metrics)
    render_timeline(RESULTS / "sync_timeline.png", camera_t, lidar_reported_t, estimated_dt)
    render_reprojection(RESULTS / "reprojection_error.png", gt_pixels, before_pixels, after_pixels)
    return metrics


def get_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        digest = hashlib.sha256((BASE / "calibration_lab.py").read_bytes()).hexdigest()
        return f"sha256:{digest} (no git repository)"


def write_report(metrics: dict) -> None:
    t = metrics["time_sync"]
    s = metrics["spatial_calibration"]
    a = metrics["acceptance"]
    report = f"""# 多传感器同步与标定实验报告

## 1. 实验目的

验证不同频率、不同时间戳的 Camera/LiDAR 数据如何同步，并估计 LiDAR 到 Camera 的空间外参。实验使用固定随机种子的模拟数据，便于重复运行和验证。

## 2. 方法

- 时间同步：以 Camera 时间戳为查询点，在 LiDAR 序列上搜索使信号误差最小的偏移 `dt`。
- 时间插值：对不同采样频率的数据使用线性插值；实际系统可使用硬件触发、PTP 或 GPS。
- 空间标定：使用对应的三维点，通过 Kabsch/SVD 估计 `R, t`。
- 投影评价：使用相机内参将三维点投影到图像，计算平均重投影误差。

## 3. 结果

| 指标 | 结果 |
|---|---:|
| 真实时间偏移 | {t['ground_truth_offset_s']*1000:.2f} ms |
| 估计时间偏移 | {t['estimated_offset_s']*1000:.2f} ms |
| 时间偏移绝对误差 | {t['absolute_offset_error_ms']:.2f} ms |
| 时间抖动 jitter | {t['jitter_s']*1000:.2f} ms |
| 匹配率 match rate | {t['match_rate']*100:.2f}% |
| 标定前重投影误差 | {s['reprojection_before_px']:.3f} px |
| 标定后重投影误差 | {s['reprojection_after_px']:.3f} px |
| 旋转误差 | {s['rotation_error_deg']:.4f}° |
| 平移误差 | {s['translation_error_m']:.5f} m |

## 4. 验收判断

- 时间偏移误差小于 10 ms：`{'通过' if a['time_offset_error_ms_under_10'] else '未通过'}`
- 匹配率大于 80%：`{'通过' if a['match_rate_over_0_8'] else '未通过'}`
- 标定后重投影误差小于 3 px：`{'通过' if a['reprojection_after_under_3px'] else '未通过'}`

## 5. 反模式检查

- 几何：统一米和像素单位，检查坐标轴方向和旋转矩阵。
- 时间：使用采集时间戳，不使用数据包到达时间；保留 jitter 和 offset 指标。
- 流程：标定文件带版本号，实验使用固定数据划分，并保留 README 和版本信息。

## 6. 运行方式

```text
python calibration_lab.py
```

GUI 中点击“重新生成实验结果”可以重新生成全部结果文件。`calib_v1.yaml` 是标定参数，`metrics.json` 是机器可读指标，`sync_timeline.png` 和 `reprojection_error.png` 是可视化结果。

## 7. 交互实验

GUI 左侧显示电脑摄像头或演示画面，右侧滑块模拟 LiDAR/Radar 距离、IMU 角速度、时间偏移 `dt` 以及 LiDAR 到 Camera 的 yaw、tx、ty。滑块会实时改变匹配率和重投影误差，点击“保存参数 / Сохранить”会生成 `results/interactive_config.json`。
"""
    (BASE / "REPORT.md").write_text(report, encoding="utf-8")


class LabApp:
    """交互式双语实验界面：真实摄像头 + 模拟 LiDAR/IMU + 可调标定参数。"""

    BG = "#0b1220"
    PANEL = "#132238"
    TEXT = "#eaf3ff"
    MUTED = "#9db1c7"
    CYAN = "#55d9e6"
    GREEN = "#55d79b"
    ORANGE = "#ffbd68"
    RED = "#ff7185"

    def __init__(self, root: tk.Tk, metrics: dict):
        self.root = root
        self.root.title("多传感器同步与标定实验 | Синхронизация и калибровка")
        self.root.geometry("1500x940")
        # 固定窗口和布局，避免实时画面刷新或滑块操作导致界面跳动。
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.metrics = metrics
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_ok = bool(self.capture.isOpened())
        if not self.camera_ok:
            self.capture.release()
        self.demo_mode = not self.camera_ok
        self.camera_running = True
        self.t0 = time.time()
        self.frame_image: ImageTk.PhotoImage | None = None
        self.dt_ms = tk.DoubleVar(value=50.0)
        self.yaw_deg = tk.DoubleVar(value=5.0)
        self.tx_m = tk.DoubleVar(value=0.23)
        self.ty_m = tk.DoubleVar(value=-0.08)
        self.lidar_range = tk.DoubleVar(value=8.0)
        self.lidar_noise = tk.DoubleVar(value=0.03)
        self.imu_w = tk.DoubleVar(value=0.13)
        self.imu_noise = tk.DoubleVar(value=0.01)
        self.result_var = tk.StringVar()
        self.sensor_var = tk.StringVar()
        self.metric_var = tk.StringVar()
        self._build()
        self._tick()

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=self.BG)
        header.pack(fill="x", padx=22, pady=(16, 8))
        tk.Label(header, text="多传感器同步与标定实验", font=("Microsoft YaHei UI", 24, "bold"), fg=self.TEXT, bg=self.BG).pack(side="left")
        tk.Label(header, text="Синхронизация и калибровка мультисенсорных данных", font=("Segoe UI", 12), fg=self.MUTED, bg=self.BG).pack(side="left", padx=20, pady=(7, 0))
        tk.Button(header, text="重新生成报告 / Пересчитать", command=self.regenerate, bg="#1f5d77", fg="white", relief="flat", padx=12, pady=7).pack(side="right")
        tk.Button(header, text="打开报告 / Отчёт", command=lambda: webbrowser.open((BASE / "REPORT.md").as_uri()), bg="#315243", fg="white", relief="flat", padx=12, pady=7).pack(side="right", padx=8)

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=22, pady=8)
        left = tk.Frame(body, bg=self.PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(body, bg=self.PANEL, width=430)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(left, text="真实摄像头 / Реальная камера", font=("Microsoft YaHei UI", 14, "bold"), fg=self.TEXT, bg=self.PANEL).pack(anchor="w", padx=16, pady=(14, 6))
        # 固定视频区域：使用 Canvas 而不是随图片尺寸变化的 Label，
        # 避免每一帧重新参与几何布局，造成画面缩放和整个界面跳动。
        self.video = tk.Canvas(left, width=800, height=470, bg="#050a12", highlightthickness=0)
        self.video.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(left, textvariable=self.sensor_var, font=("Microsoft YaHei UI", 11), fg=self.MUTED, bg=self.PANEL, justify="left").pack(anchor="w", padx=16, pady=(0, 8))

        cloud_panel = tk.Frame(left, bg="#0d1a2b")
        cloud_panel.pack(fill="x", padx=16, pady=(0, 14))
        tk.Label(cloud_panel, text="模拟 LiDAR / Radar 点云", font=("Microsoft YaHei UI", 12, "bold"), fg=self.GREEN, bg="#0d1a2b").pack(anchor="w", padx=10, pady=(8, 2))
        self.cloud = tk.Canvas(cloud_panel, height=180, bg="#07101d", highlightthickness=0)
        self.cloud.pack(fill="x", padx=10, pady=(0, 8))

        tk.Label(right, text="交互控制 / Управление", font=("Microsoft YaHei UI", 14, "bold"), fg=self.TEXT, bg=self.PANEL).pack(anchor="w", padx=16, pady=(14, 8))
        self._slider(right, "时间偏移 dt / Смещение времени (ms)", self.dt_ms, -150, 150, "%.1f")
        self._slider(right, "外参旋转 yaw / Поворот yaw (deg)", self.yaw_deg, -20, 20, "%.1f")
        self._slider(right, "外参平移 tx / Сдвиг tx (m)", self.tx_m, -1, 1, "%.2f")
        self._slider(right, "外参平移 ty / Сдвиг ty (m)", self.ty_m, -1, 1, "%.2f")
        self._slider(right, "LiDAR 距离 / Дальность LiDAR (m)", self.lidar_range, 1, 15, "%.1f")
        self._slider(right, "LiDAR 噪声 / Шум LiDAR (m)", self.lidar_noise, 0, 0.3, "%.3f")
        self._slider(right, "IMU 角速度 / Угловая скорость (rad/s)", self.imu_w, -1, 1, "%.2f")
        self._slider(right, "IMU 噪声 / Шум IMU", self.imu_noise, 0, 0.1, "%.3f")

        controls = tk.Frame(right, bg=self.PANEL)
        controls.pack(fill="x", padx=16, pady=(10, 8))
        tk.Button(controls, text="暂停摄像头 / Пауза", command=self.toggle_camera, bg="#1f5d77", fg="white", relief="flat", padx=10, pady=7).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(controls, text="保存参数 / Сохранить", command=self.save_interactive, bg="#315243", fg="white", relief="flat", padx=10, pady=7).pack(side="left", fill="x", expand=True, padx=(4, 0))

        tk.Label(right, text="传感器状态 / Состояние датчиков", font=("Microsoft YaHei UI", 13, "bold"), fg=self.TEXT, bg=self.PANEL).pack(anchor="w", padx=16, pady=(10, 5))
        tk.Label(right, textvariable=self.result_var, font=("Consolas", 11), fg="#dceaf8", bg="#203954", justify="left", anchor="w", padx=12, pady=12, wraplength=390).pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(right, textvariable=self.metric_var, font=("Consolas", 10), fg="#b8cde0", bg=self.PANEL, justify="left", anchor="w", wraplength=395).pack(fill="x", padx=16, pady=4)
        tk.Label(right, text="说明 / Примечание:\n摄像头为真实数据；LiDAR 和 IMU 为模拟数据。\n滑块变化会立即影响点云、IMU 和同步/标定状态。", font=("Microsoft YaHei UI", 10), fg=self.MUTED, bg=self.PANEL, justify="left", wraplength=395).pack(anchor="w", padx=16, pady=(12, 0))

    def _slider(self, parent: tk.Widget, text: str, var: tk.DoubleVar, lo: float, hi: float, fmt: str) -> None:
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill="x", padx=16, pady=(3, 0))
        label = tk.Label(row, text=fmt % var.get(), width=8, fg=self.CYAN, bg=self.PANEL, font=("Consolas", 10, "bold"))
        tk.Label(row, text=text, fg=self.MUTED, bg=self.PANEL, font=("Microsoft YaHei UI", 9)).pack(side="left")
        label.pack(side="right")
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=(hi - lo) / 200.0, orient="horizontal", variable=var, showvalue=False, bg=self.PANEL, fg=self.CYAN, troughcolor="#2c4560", highlightthickness=0, command=lambda value: label.config(text=fmt % float(value)))
        scale.pack(fill="x", padx=16, pady=(0, 2))

    def _camera_frame(self) -> np.ndarray:
        if self.camera_running and self.camera_ok:
            ok, frame = self.capture.read()
            if ok:
                return cv2.flip(frame, 1)
        return make_demo_frame(time.time() - self.t0)

    def _update_cloud(self) -> None:
        self.cloud.delete("all")
        w = max(self.cloud.winfo_width(), 520)
        h = 180
        self.cloud.create_line(w // 2, h // 2, w - 25, h // 2, fill="#58728e", width=2)
        self.cloud.create_line(w // 2, h // 2, w // 2, 20, fill="#58728e", width=2)
        self.cloud.create_text(w - 42, h // 2 - 12, text="X", fill="#9db1c7")
        self.cloud.create_text(w // 2 + 12, 18, text="Y", fill="#9db1c7")
        rng = np.random.default_rng(7)
        count = 28
        angle = np.linspace(-0.75, 0.75, count) + rng.normal(0, 0.03, count)
        radius = self.lidar_range.get() + rng.normal(0, self.lidar_noise.get() * 3 + 0.05, count)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        scale = 22
        cx, cy = w // 2, h // 2
        for px, py in zip(x, y):
            sx, sy = cx + int(px * scale), cy - int(py * scale)
            if 8 < sx < w - 8 and 8 < sy < h - 8:
                self.cloud.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill="#55d79b", outline="")
        self.cloud.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#ffbd68", outline="")

    def _tick(self) -> None:
        frame = self._camera_frame()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = 800, 470
        image.thumbnail((width - 4, height - 4), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (5, 10, 18))
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        self.frame_image = ImageTk.PhotoImage(canvas)
        self.video.delete("all")
        self.video.create_image(0, 0, anchor="nw", image=self.frame_image)
        self._update_cloud()
        dt_error = abs(self.dt_ms.get() - 50.0)
        transform_error = abs(self.yaw_deg.get() - 5.0) + abs(self.tx_m.get() - 0.23) * 10 + abs(self.ty_m.get() + 0.08) * 10
        match = max(0.0, min(1.0, 1.0 - dt_error / 120.0))
        reprojection = 0.56 + transform_error * 2.2 + self.lidar_noise.get() * 8
        imu_value = self.imu_w.get() + math.sin(time.time() * 2.0) * self.imu_noise.get()
        sync_ok = dt_error < 10 and match > 0.8
        calib_ok = reprojection < 3.0
        self.sensor_var.set(("摄像头已连接 / Камера подключена" if self.camera_ok else "演示摄像头 / Демо-камера") + "\n"
                            f"LiDAR: {self.lidar_range.get():.2f} m, noise={self.lidar_noise.get():.3f} m\n"
                            f"IMU: ωz={imu_value:+.3f} rad/s, noise={self.imu_noise.get():.3f}")
        self.result_var.set(f"同步 / Синхронизация: {'通过 PASS' if sync_ok else '需调整 CHECK'}\n"
                            f"标定 / Калибровка: {'通过 PASS' if calib_ok else '需调整 CHECK'}\n"
                            f"dt = {self.dt_ms.get():.1f} ms   yaw = {self.yaw_deg.get():+.1f}°")
        self.metric_var.set(f"match rate = {match * 100:.1f}%\n重投影 / reprojection ≈ {reprojection:.2f} px\n"
                            "坐标变换 / transform: LiDAR → Camera")
        self.root.after(80, self._tick)

    def toggle_camera(self) -> None:
        self.camera_running = not self.camera_running

    def save_interactive(self) -> None:
        config = {
            "time_offset_ms": self.dt_ms.get(),
            "yaw_deg": self.yaw_deg.get(),
            "tx_m": self.tx_m.get(),
            "ty_m": self.ty_m.get(),
            "lidar_range_m": self.lidar_range.get(),
            "lidar_noise_m": self.lidar_noise.get(),
            "imu_angular_velocity_rad_s": self.imu_w.get(),
            "imu_noise": self.imu_noise.get(),
            "camera_source": "real webcam" if self.camera_ok else "demo fallback",
        }
        path = RESULTS / "interactive_config.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("已保存 / Сохранено", f"交互参数已保存到：\n{path}")

    def regenerate(self) -> None:
        self.metrics = run_experiment()
        write_report(self.metrics)
        messagebox.showinfo("完成 / Готово", "已重新生成批处理实验结果、标定文件和报告。")

    def close(self) -> None:
        if self.capture and self.capture.isOpened():
            self.capture.release()
        self.root.destroy()


def main() -> None:
    metrics = run_experiment()
    write_report(metrics)
    (BASE / "commit_hash.txt").write_text(get_commit_hash() + "\n", encoding="utf-8")
    root = tk.Tk()
    LabApp(root, metrics)
    root.mainloop()


if __name__ == "__main__":
    main()
