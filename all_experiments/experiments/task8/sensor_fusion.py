"""多传感器融合演示：摄像头 + 模拟雷达 + 模拟超声波。

课程项目：异构传感器数据融合。
摄像头是真实输入；雷达和超声波用滑块模拟，便于在没有实体传感器时完成实验。
"""

from __future__ import annotations

import json
import math
import queue
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def make_demo_frame(t: float, width: int = 720, height: int = 405) -> np.ndarray:
    """摄像头不可用时的动态画面，保证课堂演示可以运行。"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 36, 50)
    cv2.rectangle(frame, (18, 18), (width - 18, height - 18), (50, 68, 86), 2)
    x = int(width * (0.5 + 0.28 * math.sin(t * 0.8)))
    y = int(height * (0.5 + 0.18 * math.cos(t * 0.6)))
    radius = int(35 + 8 * math.sin(t))
    cv2.circle(frame, (x, y), radius, (76, 194, 255), -1)
    cv2.circle(frame, (x, y), radius + 4, (170, 225, 255), 2)
    cv2.putText(frame, "DEMO CAMERA", (32, height - 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (196, 214, 232), 2, cv2.LINE_AA)
    return frame


class FusionEngine:
    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None

    def camera_measurement(self, frame: np.ndarray) -> tuple[bool, float, str]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        center = small[18:72, 32:128]
        edges = cv2.Canny(center, 60, 140)
        edge_score = float(np.mean(edges) / 255.0)
        motion_score = 0.0
        if self.previous_gray is not None:
            motion_score = float(np.mean(cv2.absdiff(small, self.previous_gray)) / 255.0)
        self.previous_gray = small
        # 画面中存在边缘或运动时，认为摄像头观察到候选目标。
        confidence = clamp(0.25 + edge_score * 1.4 + motion_score * 4.0, 0.05, 0.98)
        present = confidence >= 0.42
        reason = f"边缘 {edge_score:.2f} / 运动 {motion_score:.2f}"
        return present, confidence, reason

    @staticmethod
    def fuse(camera: tuple[bool, float], radar: tuple[bool, float], ultrasonic: tuple[bool, float], method: str) -> tuple[bool, float, str]:
        measurements = [camera, radar, ultrasonic]
        yes = sum(1 for present, _ in measurements if present)
        if method == "多数投票 / Majority vote":
            score = yes / 3.0
            return yes >= 2, score, f"{yes}/3 个传感器判断为存在"
        if method == "加权融合 / Weighted":
            weights = np.array([0.40, 0.40, 0.20], dtype=float)
            score = float(sum(w * (conf if present else 1.0 - conf) for w, (present, conf) in zip(weights, measurements)))
            return score >= 0.5, score, "权重：摄像头 0.40，雷达 0.40，超声波 0.20"
        # 简化贝叶斯：把每次观测转成似然，依次更新目标存在的概率。
        p = 0.50
        for present, conf in measurements:
            conf = clamp(conf, 0.51, 0.99)
            likelihood_if_yes = conf if present else 1.0 - conf
            likelihood_if_no = 1.0 - likelihood_if_yes
            numerator = p * likelihood_if_yes
            denominator = numerator + (1.0 - p) * likelihood_if_no
            p = numerator / denominator if denominator > 1e-9 else p
        return p >= 0.5, p, "先验 0.50，按摄像头→雷达→超声波顺序更新"


class SensorFusionApp:
    BG = "#0b1220"
    PANEL = "#111c2e"
    PANEL2 = "#17263d"
    TEXT = "#e8f0fa"
    MUTED = "#9db1c7"
    CYAN = "#50d6e6"
    GREEN = "#49d69a"
    ORANGE = "#ffb454"
    RED = "#ff7185"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("多传感器融合实验 · Sensor Fusion Lab")
        self.root.geometry("1420x900")
        self.root.minsize(1180, 760)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_ok = bool(self.capture.isOpened())
        if not self.camera_ok:
            self.capture.release()
        self.demo_mode = not self.camera_ok
        self.engine = FusionEngine()
        self.latest_frame = make_demo_frame(0)
        self.latest_rgb: Image.Image | None = None
        self.running = True
        self.history: list[dict] = []

        self.method_var = tk.StringVar(value="加权融合 / Weighted")
        self.radar_distance = tk.DoubleVar(value=2.3)
        self.radar_speed = tk.DoubleVar(value=0.8)
        self.radar_conf = tk.DoubleVar(value=0.90)
        self.ultra_distance = tk.DoubleVar(value=0.9)
        self.ultra_conf = tk.DoubleVar(value=0.85)
        self.threshold = tk.DoubleVar(value=0.42)
        self.status_var = tk.StringVar()
        self.result_var = tk.StringVar(value="等待融合结果")
        self.camera_var = tk.StringVar(value="摄像头初始化中")
        self.radar_var = tk.StringVar()
        self.ultra_var = tk.StringVar()
        self.details_var = tk.StringVar()

        self._build_style()
        self._build_ui()
        self._tick()

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Fusion.TButton", font=("Microsoft YaHei UI", 12, "bold"), padding=(14, 10), background=self.PANEL2, foreground=self.TEXT)
        style.map("Fusion.TButton", background=[("active", "#24466a")])
        # ttk.Scale 会自动使用 Horizontal 前缀；直接继承系统布局可避免
        # “Layout Horizontal.Fusion.TScale not found”错误。
        style.configure("Horizontal.TScale", background=self.PANEL, troughcolor="#263a55")
        style.configure("Fusion.TCombobox", font=("Microsoft YaHei UI", 12), fieldbackground=self.PANEL2, foreground=self.TEXT)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=self.BG)
        header.pack(fill="x", padx=24, pady=(18, 8))
        tk.Label(header, text="多传感器融合实验", font=("Microsoft YaHei UI", 25, "bold"), fg=self.TEXT, bg=self.BG).pack(side="left")
        tk.Label(header, text="Sensor Fusion Lab  ·  Camera + Radar + Ultrasonic", font=("Segoe UI", 13), fg=self.MUTED, bg=self.BG).pack(side="left", padx=22, pady=(8, 0))
        tk.Button(header, text="保存实验记录", command=self.save_log, font=("Microsoft YaHei UI", 11, "bold"), bg="#1c4f66", fg=self.TEXT, activebackground="#277795", relief="flat", padx=14, pady=8).pack(side="right")

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=24, pady=8)
        left = tk.Frame(body, bg=self.PANEL, width=770)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(body, bg=self.PANEL, width=430)
        right.pack(side="right", fill="y")
        left.pack_propagate(False)
        right.pack_propagate(False)

        tk.Label(left, text="实时视觉输入 / Live camera", font=("Microsoft YaHei UI", 14, "bold"), fg=self.TEXT, bg=self.PANEL).pack(anchor="w", padx=18, pady=(16, 8))
        self.video_label = tk.Label(left, bg="#070c15", relief="flat")
        self.video_label.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self.status_label = tk.Label(left, textvariable=self.status_var, font=("Microsoft YaHei UI", 11), fg=self.MUTED, bg=self.PANEL)
        self.status_label.pack(anchor="w", padx=18, pady=(0, 16))

        self._section(right, "融合设置 / Fusion settings")
        tk.Label(right, text="融合方法", fg=self.MUTED, bg=self.PANEL, font=("Microsoft YaHei UI", 11)).pack(anchor="w", padx=18, pady=(6, 2))
        combo = ttk.Combobox(right, textvariable=self.method_var, values=["多数投票 / Majority vote", "加权融合 / Weighted", "贝叶斯更新 / Bayesian"], state="readonly", style="Fusion.TCombobox")
        combo.pack(fill="x", padx=18, pady=(0, 14))

        self._slider(right, "雷达距离 / Radar range (m)", self.radar_distance, 0.5, 10.0, 0.1)
        self._slider(right, "雷达速度 / Radar speed (m/s)", self.radar_speed, -5.0, 5.0, 0.1)
        self._slider(right, "雷达置信度 / Radar confidence", self.radar_conf, 0.5, 0.99, 0.01)
        self._slider(right, "超声波距离 / Ultrasonic range (m)", self.ultra_distance, 0.1, 5.0, 0.1)
        self._slider(right, "超声波置信度 / Ultrasonic confidence", self.ultra_conf, 0.5, 0.99, 0.01)
        self._slider(right, "视觉阈值 / Vision threshold", self.threshold, 0.2, 0.8, 0.01)

        button_row = tk.Frame(right, bg=self.PANEL)
        button_row.pack(fill="x", padx=18, pady=(12, 12))
        tk.Button(button_row, text="开始/暂停摄像头", command=self.toggle_camera, font=("Microsoft YaHei UI", 11, "bold"), bg="#1c4f66", fg=self.TEXT, activebackground="#277795", relief="flat", padx=8, pady=8).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(button_row, text="截取当前帧", command=self.save_frame, font=("Microsoft YaHei UI", 11, "bold"), bg="#315042", fg=self.TEXT, activebackground="#3e705d", relief="flat", padx=8, pady=8).pack(side="left", fill="x", expand=True, padx=(6, 0))

        self._section(right, "传感器输出 / Sensor outputs")
        self._sensor_card(right, "摄像头 / Camera", self.camera_var, self.CYAN)
        self._sensor_card(right, "雷达 / Radar", self.radar_var, self.GREEN)
        self._sensor_card(right, "超声波 / Ultrasonic", self.ultra_var, self.ORANGE)

        result = tk.Frame(right, bg="#203954", highlightthickness=1, highlightbackground="#3d6387")
        result.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(result, text="融合结论 / Fusion result", fg=self.MUTED, bg="#203954", font=("Microsoft YaHei UI", 11)).pack(anchor="w", padx=14, pady=(12, 4))
        tk.Label(result, textvariable=self.result_var, fg=self.TEXT, bg="#203954", font=("Microsoft YaHei UI", 19, "bold"), wraplength=360, justify="left").pack(anchor="w", padx=14, pady=(0, 8))
        tk.Label(result, textvariable=self.details_var, fg="#bcd1e6", bg="#203954", font=("Microsoft YaHei UI", 10), wraplength=360, justify="left").pack(anchor="w", padx=14, pady=(0, 14))

        tk.Label(right, text="项目说明：摄像头为真实输入；雷达和超声波为可调模拟传感器。\n适合演示高层决策融合与加权/贝叶斯方法。", fg=self.MUTED, bg=self.PANEL, font=("Microsoft YaHei UI", 10), justify="left", wraplength=390).pack(anchor="w", padx=18, pady=(10, 0))

    def _section(self, parent: tk.Widget, title: str) -> None:
        tk.Label(parent, text=title, fg=self.TEXT, bg=self.PANEL, font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w", padx=18, pady=(16, 4))

    def _slider(self, parent: tk.Widget, text: str, variable: tk.DoubleVar, lo: float, hi: float, step: float) -> None:
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill="x", padx=18, pady=(4, 2))
        value_label = tk.Label(row, text=f"{variable.get():.2f}", width=7, fg=self.CYAN, bg=self.PANEL, font=("Consolas", 10, "bold"))
        tk.Label(row, text=text, fg=self.MUTED, bg=self.PANEL, font=("Microsoft YaHei UI", 10)).pack(side="left")
        value_label.pack(side="right")
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=variable, style="Horizontal.TScale", command=lambda v: value_label.config(text=f"{float(v):.2f}"))
        scale.pack(fill="x", padx=18, pady=(0, 5))

    def _sensor_card(self, parent: tk.Widget, title: str, variable: tk.StringVar, color: str) -> None:
        card = tk.Frame(parent, bg=self.PANEL2, highlightthickness=1, highlightbackground="#29435f")
        card.pack(fill="x", padx=18, pady=4)
        tk.Frame(card, bg=color, width=6).pack(side="left", fill="y")
        tk.Label(card, text=title, fg=self.TEXT, bg=self.PANEL2, font=("Microsoft YaHei UI", 11, "bold"), width=16, anchor="w").pack(side="left", padx=10, pady=9)
        tk.Label(card, textvariable=variable, fg="#c5d8ea", bg=self.PANEL2, font=("Microsoft YaHei UI", 10), anchor="w", justify="left", wraplength=245).pack(side="left", fill="x", expand=True, pady=8)

    def _tick(self) -> None:
        if self.running:
            if self.capture and self.capture.isOpened():
                ok, frame = self.capture.read()
                if ok:
                    self.latest_frame = cv2.flip(frame, 1)
            else:
                self.latest_frame = make_demo_frame(time.time())
            self._update()
        self.root.after(80, self._tick)

    def _update(self) -> None:
        frame = self.latest_frame.copy()
        cam_present, cam_conf, cam_reason = self.engine.camera_measurement(frame)
        # 用滑块模拟其他传感器的判断：距离越小，越容易判断为存在目标。
        radar_present = self.radar_distance.get() <= 6.0
        ultra_present = self.ultra_distance.get() <= 1.6
        final, score, detail = FusionEngine.fuse(
            (cam_present, max(cam_conf, self.threshold.get())),
            (radar_present, self.radar_conf.get()),
            (ultra_present, self.ultra_conf.get()),
            self.method_var.get(),
        )
        cv2.putText(frame, f"CAMERA {'OBJECT' if cam_present else 'CLEAR'}  {cam_conf:.2f}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 215, 230), 2, cv2.LINE_AA)
        cv2.putText(frame, f"RADAR {self.radar_distance.get():.1f}m", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 230, 170), 2, cv2.LINE_AA)
        cv2.putText(frame, f"ULTRASONIC {self.ultra_distance.get():.1f}m", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 190, 90), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FUSION {'OBJECT' if final else 'CLEAR'}  {score:.2f}", (28, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 150) if final else (255, 115, 135), 2, cv2.LINE_AA)
        self._show_frame(frame)

        self.camera_var.set(f"{'发现候选目标' if cam_present else '未发现明显目标'} · 置信度 {cam_conf:.2f}\n{cam_reason}")
        self.radar_var.set(f"{'有目标' if radar_present else '无目标'} · 距离 {self.radar_distance.get():.1f} m · 速度 {self.radar_speed.get():+.1f} m/s")
        self.ultra_var.set(f"{'近距离障碍物' if ultra_present else '未检测到近障碍'} · 距离 {self.ultra_distance.get():.1f} m")
        self.result_var.set(f"{'存在目标 / OBJECT PRESENT' if final else '暂无目标 / NO OBJECT'}\n置信度 {score:.2f}")
        self.details_var.set(detail)
        self.status_var.set(("演示模式：未找到摄像头，正在使用动态模拟画面" if self.demo_mode else "摄像头已连接") + "   ·   融合层次：高层决策融合")
        self.history.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "method": self.method_var.get(), "camera": {"present": cam_present, "confidence": round(cam_conf, 4)}, "radar": {"present": radar_present, "distance_m": self.radar_distance.get(), "speed_mps": self.radar_speed.get()}, "ultrasonic": {"present": ultra_present, "distance_m": self.ultra_distance.get()}, "final": final, "score": round(score, 4)})
        if len(self.history) > 2000:
            self.history = self.history[-2000:]

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.latest_rgb = image.copy()
        width = max(self.video_label.winfo_width(), 400)
        height = max(self.video_label.winfo_height(), 260)
        image.thumbnail((width - 4, height - 4), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (7, 12, 21))
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        photo = ImageTk.PhotoImage(canvas)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def toggle_camera(self) -> None:
        self.running = not self.running

    def save_frame(self) -> None:
        filename = RESULTS / f"camera_frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
        if self.latest_rgb is not None:
            self.latest_rgb.save(filename)
            messagebox.showinfo("已保存", f"当前帧已保存：\n{filename}")

    def save_log(self) -> None:
        if not self.history:
            messagebox.showinfo("提示", "暂时没有实验记录")
            return
        target = filedialog.asksaveasfilename(initialdir=str(RESULTS), initialfile="fusion_log.json", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if target:
            Path(target).write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.running = False
        if self.capture and self.capture.isOpened():
            self.capture.release()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = SensorFusionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
