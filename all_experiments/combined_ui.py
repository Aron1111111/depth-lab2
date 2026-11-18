"""Fixed-size bilingual browser for all completed image-processing experiments."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
CATALOG = [
    ("task1", "图像退化与恢复 / Degradation", "高斯噪声、椒盐噪声和模糊的恢复指标"),
    ("task2", "图像增强 / Enhancement", "亮度、对比度和清晰度增强"),
    ("task3", "图像测量 / Measurement", "面积、周长、颜色和几何测量"),
    ("task4", "图像分割 / Segmentation", "目标区域提取和分割结果"),
    ("task5", "全景拼接 / Panorama", "两幅图像的特征匹配和拼接"),
    ("task6", "经典分类 / Classical ML", "猫狗图像的传统机器学习分类"),
    ("task7", "CNN 分类 / CNN", "CNN 与基线模型的分类指标"),
    ("task8", "多传感器融合 / Fusion", "Camera、Radar、Ultrasonic 融合演示"),
    ("task9", "同步与标定 / Calibration", "Camera、LiDAR、IMU 的同步和标定"),
    ("lab2", "深度修复 / Depth", "缺失 depth 修复与可复现性实验"),
]
CONCLUSIONS = {
    "task1": "结论：不同噪声需要不同恢复方法，PSNR 和 SSIM 可以用来比较恢复质量。/ Вывод: для разных шумов нужны разные методы восстановления; PSNR и SSIM помогают сравнить качество.",
    "task2": "结论：图像增强可以改善亮度和对比度，但参数过大可能放大噪声。/ Вывод: улучшение повышает яркость и контраст, но слишком сильные параметры усиливают шум.",
    "task3": "结论：通过阈值、轮廓和几何公式，可以从图像中得到可解释的测量结果。/ Вывод: пороговая обработка, контуры и геометрические формулы дают интерпретируемые измерения.",
    "task4": "结论：分割的关键是把目标和背景分开，并用可视化检查结果是否合理。/ Вывод: главное в сегментации — отделить объект от фона и проверить результат визуально.",
    "task5": "结论：特征匹配和单应性变换可以把两张有重叠区域的图片拼成全景图。/ Вывод: сопоставление признаков и гомография позволяют объединить изображения с перекрытием в панораму.",
    "task6": "结论：传统分类器可以完成猫狗分类，但效果依赖特征、数据划分和参数。/ Вывод: классические классификаторы решают задачу кошка/собака, но результат зависит от признаков, разбиения и параметров.",
    "task7": "结论：CNN 能够自动学习图像特征，应该和基线模型一起用指标比较。/ Вывод: CNN автоматически изучает признаки изображения, поэтому его нужно сравнивать с базовыми моделями по метрикам.",
    "task8": "结论：融合多个传感器可以提高判断稳定性，但必须处理时间和坐标的一致性。/ Вывод: слияние сенсоров повышает устойчивость, но требует согласованных времени и координат.",
    "task9": "结论：同步和标定直接影响多传感器融合；时间偏移或外参错误都会增加重投影误差。/ Вывод: синхронизация и калибровка напрямую влияют на fusion; ошибка времени или внешних параметров увеличивает reprojection error.",
    "lab2": "结论：缺失 depth 应先显式标记，再用形态学和 Guided Filter 修复，并用 valid 比例和 RMSE 验证。/ Вывод: пропуски depth нужно сначала явно отметить, затем восстановить морфологией и Guided Filter и проверить valid-процентом и RMSE.",
}


class AllExperimentsUI:
    BG = "#0b1220"
    CARD = "#111c2e"
    FG = "#e8eef7"
    MUTED = "#9fb0c5"
    BLUE = "#42a5f5"

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("图像处理实验总览 / Обзор лабораторных работ")
        root.geometry("1600x1000")
        root.resizable(False, False)
        root.configure(bg=self.BG)
        self.photos: list[ImageTk.PhotoImage] = []
        self._build()

    def _build(self):
        header = tk.Frame(self.root, bg=self.BG, height=86)
        header.pack(fill="x", padx=24, pady=(18, 8))
        tk.Label(header, text="图像处理实验总览  ·  Обзор лабораторных работ", bg=self.BG, fg=self.FG,
                 font=("Segoe UI", 25, "bold")).pack(anchor="w")
        tk.Label(header, text="点击实验标签查看对应图片、结果指标、代码和说明 / Нажмите вкладку эксперимента",
                 bg=self.BG, fg=self.MUTED, font=("Segoe UI", 11)).pack(anchor="w", pady=(5, 0))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("All.TNotebook", background=self.BG, borderwidth=0)
        style.configure("All.TNotebook.Tab", background="#dbe7f3", foreground="#35516b",
                        padding=(15, 11), font=("Segoe UI", 10))
        style.map("All.TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#173d68")])
        notebook = ttk.Notebook(self.root, style="All.TNotebook", width=1545, height=835)
        notebook.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        for key, title, subtitle in CATALOG:
            if not self._image_candidates(EXPERIMENTS / key, key):
                continue
            tab = tk.Frame(notebook, bg=self.BG)
            notebook.add(tab, text=key.upper())
            self._build_experiment(tab, key, title, subtitle)

    def _build_experiment(self, parent, key, title, subtitle):
        folder = EXPERIMENTS / key
        tk.Label(parent, text=title, bg=self.BG, fg=self.BLUE, font=("Segoe UI", 21, "bold"), anchor="w").pack(fill="x", padx=20, pady=(18, 2))
        tk.Label(parent, text=subtitle, bg=self.BG, fg=self.MUTED, font=("Segoe UI", 11), anchor="w").pack(fill="x", padx=20, pady=(0, 12))

        gallery = tk.Frame(parent, bg=self.BG, height=430)
        gallery.pack(fill="x", padx=20)
        gallery.pack_propagate(False)
        images = self._image_candidates(folder, key)
        count = max(1, len(images))
        card_width = max(350, int(1500 / count) - 35)
        canvas_width = card_width - 30
        canvas_height = 340
        for i, path in enumerate(images):
            card = tk.Frame(gallery, bg=self.CARD, width=card_width, height=410)
            card.pack(side="left", padx=(0, 12), expand=True)
            card.pack_propagate(False)
            try:
                im = Image.open(path).convert("RGB")
                im.thumbnail((canvas_width, canvas_height))
                canvas = tk.Canvas(card, width=canvas_width, height=canvas_height, bg="#050a12", highlightthickness=0)
                canvas.pack(padx=15, pady=(12, 6))
                photo = ImageTk.PhotoImage(im)
                self.photos.append(photo)
                canvas.image = photo
                canvas.create_image(0, 0, anchor="nw", image=photo)
                canvas.bind("<Button-1>", lambda _event, image_path=path: self._open_image(image_path))
                canvas.configure(cursor="hand2")
                canvas.update_idletasks()
                text = path.name
            except Exception:
                text = "无法读取图片 / Не удалось открыть"
            tk.Label(card, text=text, bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 9), wraplength=320).pack(padx=12)

    def _open_image(self, path: Path):
        """Open the selected experiment image in a larger, fixed preview window."""
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return
        preview = tk.Toplevel(self.root)
        preview.title(f"图片预览 / Просмотр · {path.name}")
        preview.geometry("1200x820")
        preview.resizable(False, False)
        preview.configure(bg=self.BG)
        tk.Label(preview, text=f"{path.parent.name}  ·  {path.name}", bg=self.BG, fg=self.FG,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18, pady=(14, 8))
        canvas = tk.Canvas(preview, width=1160, height=730, bg="#050a12", highlightthickness=0)
        canvas.pack(padx=20, pady=(0, 16))
        image.thumbnail((1140, 710))
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo
        canvas.create_image(580, 365, anchor="center", image=photo)

        lower = tk.Frame(parent, bg=self.CARD, height=365)
        lower.pack(fill="both", expand=True, padx=20, pady=(14, 20))
        lower.pack_propagate(False)
        conclusion = tk.Frame(lower, bg="#10243a", height=66)
        conclusion.pack(fill="x", padx=18, pady=(14, 0))
        conclusion.pack_propagate(False)
        tk.Label(conclusion, text="实验结论 / Вывод", bg="#10243a", fg=self.BLUE,
                 font=("Segoe UI", 11, "bold"), anchor="w").pack(side="left", padx=(12, 16), pady=10)
        tk.Label(conclusion, text=CONCLUSIONS.get(key, "暂无结论 / Вывод пока не добавлен"), bg="#10243a", fg=self.FG,
                 font=("Segoe UI", 9), anchor="w", justify="left", wraplength=1180).pack(side="left", fill="both", expand=True, pady=8)
        metrics = self._metrics(folder)
        left = tk.Frame(lower, bg=self.CARD, width=720)
        left.pack(side="left", fill="both", expand=True, padx=18, pady=14)
        tk.Label(left, text="指标 / Метрики", bg=self.CARD, fg=self.BLUE, font=("Segoe UI", 14, "bold"), anchor="w").pack(fill="x")
        metric_text = tk.Text(left, bg="#0d1727", fg=self.FG, relief="flat", font=("Consolas", 10), wrap="word")
        metric_text.pack(fill="both", expand=True, pady=(8, 0))
        metric_text.insert("1.0", metrics)
        metric_text.config(state="disabled")
        right = tk.Frame(lower, bg=self.CARD, width=720)
        right.pack(side="left", fill="both", expand=True, padx=18, pady=14)
        tk.Label(right, text="文件与运行 / Файлы и запуск", bg=self.CARD, fg=self.BLUE, font=("Segoe UI", 14, "bold"), anchor="w").pack(fill="x")
        files = sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.exists() else []
        text = "文件 / Файлы:\n" + "\n".join(f"· {x}" for x in files[:30])
        text += "\n\n结果目录 / Папка результатов: " + ("results/" if (folder / "results").exists() else "无独立 results")
        box = tk.Text(right, bg="#0d1727", fg=self.MUTED, relief="flat", font=("Consolas", 10), wrap="word")
        box.pack(fill="both", expand=True, pady=(8, 0)); box.insert("1.0", text); box.config(state="disabled")

    def _image_candidates(self, folder: Path, key: str):
        preferred = {
            "task1": ["car.jpg", "car_results/original.png", "car_results/gaussian_noise_degraded.png", "car_results/gaussian_noise_best_psnr.png"],
            "task2": ["results_zh_ru.png"], "task3": ["cat.jpg", "results_zh_ru_p4.png"], "task4": ["dog.jpg"],
            "task5": ["1.png", "2.png", "panorama.png"], "lab2": ["baseball_sample.png", "results/depth_corrupted.png", "results/depth_restored.png", "results/validity_mask.png"],
        }
        paths = [folder / x for x in preferred.get(key, []) if (folder / x).exists()]
        if not paths:
            paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.stat().st_size < 15_000_000])[:4]
        return paths[:4]

    def _metrics(self, folder: Path) -> str:
        candidates = list(folder.rglob("*.json"))
        lines = []
        for path in candidates[:3]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                lines.append(f"[{path.name}]\n{json.dumps(data, ensure_ascii=False, indent=2)[:4500]}")
            except Exception:
                pass
        if lines:
            return "\n\n".join(lines)
        return "该实验没有独立 JSON 指标文件。\nНет отдельного JSON-файла с метриками.\n\n请查看对应 README、REPORT 或程序输出。\nСм. README, REPORT или вывод программы."


if __name__ == "__main__":
    root = tk.Tk()
    AllExperimentsUI(root)
    root.mainloop()
