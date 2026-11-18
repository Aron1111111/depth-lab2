"""Fixed-size bilingual interactive UI for Laboratory 2."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from restore_depth import RESULTS, SEED, guided_filter, morph_close


class DepthLabUI:
    BG = "#0b1220"
    CARD = "#111c2e"
    FG = "#e8eef7"
    MUTED = "#9fb0c5"
    BLUE = "#42a5f5"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("实验二 / Лаборатория 2 · Depth Restoration")
        root.geometry("1500x940")
        root.resizable(False, False)
        root.configure(bg=self.BG)
        self.images: list[ImageTk.PhotoImage] = []
        self.metrics: dict = {}
        self.baseball = Image.open(Path(__file__).with_name("baseball_sample.png")).convert("RGB").resize((260, 180))
        self._build()
        self.run_experiment()

    def label(self, parent, text, size=11, color=None, bold=False):
        return tk.Label(parent, text=text, font=("Segoe UI", size, "bold" if bold else "normal"),
                        fg=color or self.FG, bg=self.CARD, anchor="w")

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=self.BG, height=82)
        header.pack(fill="x", padx=24, pady=(18, 8))
        tk.Label(header, text="深度数据恢复实验  ·  Восстановление depth", font=("Segoe UI", 25, "bold"),
                 fg=self.FG, bg=self.BG).pack(anchor="w")
        tk.Label(header, text="固定随机种子 · 掩码 · 形态学闭运算 · Guided Filter · 指标对比",
                 font=("Segoe UI", 11), fg=self.MUTED, bg=self.BG).pack(anchor="w", pady=(5, 0))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Depth.TNotebook", background=self.BG, borderwidth=0)
        style.configure("Depth.TNotebook.Tab", background="#dbe7f3", foreground="#35516b", padding=(24, 12), font=("Segoe UI", 11))
        style.map("Depth.TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#173d68")])
        notebook = ttk.Notebook(self.root, style="Depth.TNotebook", width=1452, height=770)
        notebook.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        depth_tab = tk.Frame(notebook, bg=self.BG)
        notebook.add(depth_tab, text="Глубина / 深度")
        metrics_tab = tk.Frame(notebook, bg=self.CARD)
        notebook.add(metrics_tab, text="Метрики / 指标")
        data_tab = tk.Frame(notebook, bg=self.CARD)
        notebook.add(data_tab, text="Data card")
        readme_tab = tk.Frame(notebook, bg=self.CARD)
        notebook.add(readme_tab, text="README")
        checklist_tab = tk.Frame(notebook, bg=self.CARD)
        notebook.add(checklist_tab, text="Чек-лист / 清单")
        questions_tab = tk.Frame(notebook, bg=self.CARD)
        notebook.add(questions_tab, text="Вопросы / 问题")

        controls = tk.Frame(depth_tab, bg=self.CARD, width=1452, height=128)
        controls.pack(fill="x", pady=(0, 12))
        controls.pack_propagate(False)
        self.missing_var = tk.DoubleVar(value=13.3)
        self.hole_var = tk.IntVar(value=28)
        self.radius_var = tk.IntVar(value=4)
        self._slider(controls, "缺失率 / Пропуски (%)", self.missing_var, 2, 35, 0, 10)
        self._slider(controls, "大孔洞边长 / Размер дыры", self.hole_var, 8, 55, 1, 10)
        self._slider(controls, "Guided 半径 / Радиус", self.radius_var, 1, 10, 2, 10)
        tk.Button(controls, text="重新运行 / Запустить", command=self.run_experiment, bg=self.BLUE, fg="white",
                  relief="flat", font=("Segoe UI", 11, "bold"), width=18, cursor="hand2").place(x=1160, y=43, height=42)

        body = tk.Frame(depth_tab, bg=self.BG)
        body.pack(fill="both", expand=True, padx=24)
        self.canvases = []
        for title in ["棒球图像 / Baseball RGB", "缺失深度 / Corrupted", "修复深度 / Restored", "有效性掩码 / Mask"]:
            card = tk.Frame(body, bg=self.CARD, width=350, height=390)
            card.pack(side="left", padx=(0, 12), pady=(0, 12))
            card.pack_propagate(False)
            self.label(card, title, 13, self.BLUE, True).pack(fill="x", padx=14, pady=(12, 8))
            canvas = tk.Canvas(card, width=322, height=300, bg="#050a12", highlightthickness=0)
            canvas.pack(padx=14)
            self.canvases.append(canvas)

        bottom = tk.Frame(depth_tab, bg=self.CARD, height=220)
        bottom.pack(fill="x", padx=24, pady=(0, 18))
        bottom.pack_propagate(False)
        self.status = tk.Label(bottom, text="", font=("Segoe UI", 12), fg=self.FG, bg=self.CARD, anchor="w", justify="left")
        self.status.pack(fill="x", padx=16, pady=(14, 8))
        self.log = tk.Text(bottom, height=5, width=115, bg="#0d1727", fg=self.MUTED, relief="flat",
                           font=("Consolas", 10), state="disabled")
        self.log.pack(side="left", padx=16, pady=(0, 12))
        tk.Button(bottom, text="保存指标 / Сохранить", command=self.save_metrics, bg="#1f8f68", fg="white",
                  relief="flat", font=("Segoe UI", 10, "bold"), width=18, cursor="hand2").pack(side="right", padx=22, pady=54)

        self._doc_panel(metrics_tab, "实验指标 / Метрики", "results/metrics_before_after.json")
        self._doc_panel(data_tab, "数据卡片 / Data Card", "data_card.yaml")
        self._doc_panel(readme_tab, "实验说明 / README", "README.md")
        self._doc_panel(checklist_tab, "可复现性检查清单 / Чек-лист воспроизводимости", "CHECKLIST.md")
        self._questions_panel(questions_tab)

    def _doc_panel(self, parent, title, relative_path):
        self.label(parent, title, 18, self.BLUE, True).pack(fill="x", padx=24, pady=(20, 10))
        path = Path(__file__).resolve().parent / relative_path
        if not path.exists():
            path = Path(__file__).resolve().parent.parent / relative_path
        text = tk.Text(parent, bg="#0d1727", fg=self.FG, insertbackground=self.FG, relief="flat",
                       font=("Consolas", 11), wrap="word", padx=18, pady=14)
        text.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        text.insert("1.0", path.read_text(encoding="utf-8") if path.exists() else "文件未找到 / Файл не найден")
        text.config(state="disabled")

    def _questions_panel(self, parent):
        self.label(parent, "课程问题 / Вопросы к лабораторной", 18, self.BLUE, True).pack(fill="x", padx=24, pady=(20, 14))
        questions = [
            "1. 三种缺失类型是什么？ / Какие бывают три типа пропусков?\n答：随机缺失、数据包缺失、结构性缺失。/ Случайные, пакетные и структурные пропуски.",
            "2. 为什么 Inpainting 要最后使用？ / Почему Inpainting применяется последним?\n答：它可能生成看起来合理但不真实的数据，所以先用掩码、插值等保守方法。/ Метод может придумать правдоподобные, но неверные значения, поэтому сначала применяем маску и более безопасные методы.",
            "3. 容错的四个步骤是什么？ / Какие четыре шага отказоустойчивости?\n答：发现、隔离、重新配置、降级运行。/ Обнаружить, изолировать, реконфигурировать и деградировать.",
            "4. Data Card 应包含哪些信息？ / Что должно быть в Data Card?\n答：数据集、采集环境、传感器频率、标定、单位、坐标系、已知问题和许可证。/ Датасет, условия съёмки, сенсоры, частоты, калибровка, единицы, система координат, проблемы и лицензия.",
            "5. Reproducibility 和 Repeatability 有什么区别？ / Чем отличаются reproducibility и repeatability?\n答：Repeatability 是同一个人用相同代码和数据重复得到相同结果；Reproducibility 是换一个人也能得到相同结果。/ Repeatability — тот же человек, код и данные дают тот же результат; reproducibility — другой человек может получить тот же результат.",
        ]
        box = tk.Frame(parent, bg=self.CARD)
        box.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        for q in questions:
            tk.Label(box, text=q, bg="#0d1727", fg=self.FG, anchor="w", justify="left",
                     font=("Segoe UI", 11), padx=18, pady=14).pack(fill="x", pady=5)

    def _slider(self, parent, text, variable, lo, hi, x, y):
        tk.Label(parent, text=text, bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 10), anchor="w").place(x=26 + x * 380, y=15)
        tk.Scale(parent, variable=variable, from_=lo, to=hi, orient="horizontal", length=300, resolution=1,
                 bg=self.CARD, fg=self.FG, troughcolor="#263854", highlightthickness=0, showvalue=True).place(x=18 + x * 380, y=34)

    def _scene(self):
        h, w = 180, 260
        yy, xx = np.mgrid[0:h, 0:w]
        guidance = np.asarray(self.baseball.convert("L"), dtype=np.float32) / 255.0
        # Use the baseball image as an edge-rich guidance image for a synthetic depth map.
        truth = 2.0 + 0.35 * guidance + 0.002 * xx + 0.0015 * yy
        rng = np.random.default_rng(SEED)
        valid = rng.random((h, w)) >= self.missing_var.get() / 100.0
        hole = int(self.hole_var.get())
        valid[48:48 + hole, 95:95 + hole] = False
        valid[112:112 + min(hole + 2, h - 112), 170:170 + min(hole + 10, w - 170)] = False
        observed = truth.copy(); observed[~valid] = 0
        closed = morph_close(valid, 1)
        estimate = guided_filter(guidance, observed, valid, self.radius_var.get())
        estimate = np.clip(estimate, truth.min(), truth.max())
        restored = observed.copy(); restored[~valid] = estimate[~valid]
        restored[closed & ~valid] = estimate[closed & ~valid]
        return truth, observed, restored, valid | closed

    @staticmethod
    def _photo(arr, mask=False):
        if mask:
            image = (arr.astype(np.uint8) * 255)
        else:
            lo, hi = float(arr.min()), float(arr.max())
            image = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image, "L").resize((322, 300))
        return ImageTk.PhotoImage(pil.convert("RGB"))

    @staticmethod
    def _photo_rgb(image):
        return ImageTk.PhotoImage(image.resize((322, 300)))

    def run_experiment(self):
        truth, observed, restored, valid = self._scene()
        original_valid = observed > 0
        missing = ~original_valid
        self.metrics = {
            "seed": SEED,
            "missing_rate_percent": float(np.mean(missing) * 100),
            "valid_percent_before": float(np.mean(original_valid) * 100),
            "valid_percent_after": float(np.mean(valid) * 100),
            "rmse_missing_m": float(np.sqrt(np.mean((restored[missing] - truth[missing]) ** 2))),
            "rmse_all_m": float(np.sqrt(np.mean((restored - truth) ** 2))),
        }
        self.images = [self._photo_rgb(self.baseball), self._photo(observed), self._photo(restored), self._photo(valid, True)]
        for c, im in zip(self.canvases, self.images):
            c.delete("all"); c.create_image(0, 0, anchor="nw", image=im)
        self.status.config(text=(f"有效率 / Valid: {self.metrics['valid_percent_before']:.2f}% → {self.metrics['valid_percent_after']:.2f}%     "
                                f"RMSE 缺失区: {self.metrics['rmse_missing_m']:.4f} m     全图: {self.metrics['rmse_all_m']:.4f} m"))
        self._write_log("完成一次实验 / Эксперимент завершён: " + json.dumps(self.metrics, ensure_ascii=False))

    def _write_log(self, line):
        self.log.config(state="normal"); self.log.insert("end", line + "\n"); self.log.see("end"); self.log.config(state="disabled")

    def save_metrics(self):
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "interactive_metrics.json").write_text(json.dumps(self.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_log("指标已保存 / Метрики сохранены: results/interactive_metrics.json")
        messagebox.showinfo("已保存 / Готово", "交互实验指标已保存到 lab2/results。")


if __name__ == "__main__":
    app = tk.Tk()
    DepthLabUI(app)
    app.mainloop()
