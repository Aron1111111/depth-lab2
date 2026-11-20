"""A2/B2/C2 bilingual result viewer for the exam demonstration."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


class ExamUI:
    BG = "#0b1220"; CARD = "#14243b"; FG = "#eef4ff"; BLUE = "#55b6ff"; MUTED = "#b7c5d8"

    def __init__(self, root):
        self.root = root
        root.title("A2 · B2 · C2 运行结果 / Результаты запуска")
        root.geometry("1600x980"); root.resizable(False, False); root.configure(bg=self.BG)
        self.refs = []
        self.build()

    def build(self):
        tk.Label(self.root, text="试卷第二问运行结果  ·  Результаты A2 / B2 / C2", bg=self.BG, fg=self.FG,
                 font=("Segoe UI", 25, "bold")).pack(anchor="w", padx=24, pady=(18, 4))
        tk.Label(self.root, text="A2 RANSAC 全景 · B2 卡尔曼融合 · C2 形态学处理 / Панорама · Калман · Морфология",
                 bg=self.BG, fg=self.MUTED, font=("Segoe UI", 11)).pack(anchor="w", padx=24, pady=(0, 10))
        style = ttk.Style(); style.theme_use("clam")
        style.configure("Exam.TNotebook", background=self.BG, borderwidth=0)
        style.configure("Exam.TNotebook.Tab", padding=(30, 12), font=("Segoe UI", 12), background="#dbe7f3", foreground="#35516b")
        style.map("Exam.TNotebook.Tab", background=[("selected", "#ffffff")])
        nb = ttk.Notebook(self.root, style="Exam.TNotebook", width=1545, height=820); nb.pack(padx=24, pady=(0, 20))
        a = tk.Frame(nb, bg=self.BG); b = tk.Frame(nb, bg=self.BG); c = tk.Frame(nb, bg=self.BG)
        nb.add(a, text="A2 RANSAC 全景"); nb.add(b, text="B2 卡尔曼融合"); nb.add(c, text="C2 形态学")
        self.build_a2(a); self.build_b2(b); self.build_c2(c)

    def img(self, parent, path, width, height):
        box = tk.Frame(parent, bg=self.CARD, width=width + 24, height=height + 52); box.pack_propagate(False)
        box.pack(side="left", padx=12, pady=12)
        canvas = tk.Canvas(box, width=width, height=height, bg="#050a12", highlightthickness=0); canvas.pack(padx=12, pady=(12, 4))
        if path.exists():
            im = Image.open(path).convert("RGB"); im.thumbnail((width, height)); photo = ImageTk.PhotoImage(im); self.refs.append(photo)
            canvas.create_image(width//2, height//2, image=photo); canvas.image = photo
        tk.Label(box, text=path.name, bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 9)).pack()
        return box

    def build_a2(self, parent):
        tk.Label(parent, text="A2：基于 RANSAC 的全景拼接 / Панорама на основе RANSAC", bg=self.BG, fg=self.BLUE, font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=20, pady=16)
        row = tk.Frame(parent, bg=self.BG); row.pack()
        for name in ["input_left.png", "input_right.png", "panorama_ransac.png"]: self.img(row, RESULTS / name, 470, 280)
        tk.Label(parent, text="程序要点 / Основные шаги：SIFT → ratio test → Homography → RANSAC → panorama",
                 bg=self.CARD, fg=self.FG, font=("Segoe UI", 13), padx=20, pady=18).pack(fill="x", padx=20, pady=20)

    def build_b2(self, parent):
        tk.Label(parent, text="B2：卡尔曼滤波多传感器融合 / Слияние сенсоров фильтром Калмана", bg=self.BG, fg=self.BLUE, font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=20, pady=16)
        canvas = tk.Canvas(parent, width=1450, height=500, bg="#07101d", highlightthickness=0); canvas.pack(padx=35, pady=10)
        data = json.loads((RESULTS / "kalman_metrics.json").read_text(encoding="utf-8"))
        xs = [80 + i * 300 for i in range(len(data))]
        def points(key): return [(x, 420 - (row[key] - 9.5) * 150) for x, row in zip(xs, data)]
        canvas.create_text(25, 30, text="位置 / Положение", fill=self.MUTED, anchor="w", font=("Segoe UI", 11))
        canvas.create_line(60, 420, 1400, 420, fill="#60748d"); canvas.create_line(60, 60, 60, 420, fill="#60748d")
        for key, color, label in [("sensor_a", "#ffc857", "传感器A"), ("sensor_b", "#ff8c69", "传感器B"), ("fused", "#55b6ff", "卡尔曼")]:
            pts = points(key); canvas.create_line(*[v for p in pts for v in p], fill=color, width=4, smooth=True)
            canvas.create_text(1120 + ["sensor_a", "sensor_b", "fused"].index(key)*90, 35, text=label, fill=color, font=("Segoe UI", 10))
        tk.Label(parent, text="卡尔曼结果 / Результат Калмана 已保存到 results/kalman_metrics.json；融合结果逐步稳定。/ Оценка постепенно стабилизируется.", bg=self.CARD, fg=self.FG, font=("Segoe UI", 13), padx=20, pady=18).pack(fill="x", padx=20, pady=18)

    def build_c2(self, parent):
        tk.Label(parent, text="C2：形态学增强 / Морфологическая обработка", bg=self.BG, fg=self.BLUE, font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=20, pady=16)
        row = tk.Frame(parent, bg=self.BG); row.pack()
        for name in ["input.png", "erosion.png", "dilation.png", "opening.png", "closing.png", "result.png"]: self.img(row, RESULTS / name, 220, 230)
        tk.Label(parent, text="腐蚀 / Эрозия · 膨胀 / Дилатация · 开运算 / Открытие · 闭运算 / Закрытие",
                 bg=self.CARD, fg=self.FG, font=("Segoe UI", 13), padx=20, pady=18).pack(fill="x", padx=20, pady=20)


if __name__ == "__main__":
    root = tk.Tk(); ExamUI(root); root.mainloop()
