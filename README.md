# Depth Restoration Lab 2

实验二：可复现的深度数据恢复实验（中俄双语交互页面）。
Лабораторная 2: воспроизводимое восстановление depth с двуязычным интерфейсом.

## 内容

- `lab2/restore_depth.py`：恢复缺失 depth 数据；
- `lab2/depth_lab_ui.py`：固定尺寸交互 UI，可调节缺失率、孔洞大小和 Guided Filter 半径；
- `lab2/data_card.yaml`：数据卡片；
- `lab2/README.md`：实验详细说明；
- `lab2/reproduce_all.bat`：一键复现实验；
- `lab2/results/`：处理前后图像、掩码和指标。

## 快速运行

```powershell
cd lab2
..\.venv\Scripts\python.exe restore_depth.py
```

启动交互页面：

```powershell
cd lab2
run_ui.bat
```

详细要求、Data Card 和可复现性检查见 [`lab2/README.md`](lab2/README.md) 和 [`lab2/CHECKLIST.md`](lab2/CHECKLIST.md)。
