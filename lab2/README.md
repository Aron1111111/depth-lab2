# Laboratory 2 — восстановление depth / 深度修复

实验目标：对含缺失值的深度图进行恢复，并用固定数据、固定随机种子和固定依赖生成可复现结果。

## 运行

```powershell
\.venv\Scripts\python.exe lab2\restore_depth.py
```

或双击 `reproduce_all.bat`。结果写入 `lab2/results/`。

处理流程：

1. `valid = depth > 0` 生成有效性掩码；
2. 3×3 形态学闭运算处理小孔；
3. 使用带掩码统计的 Guided Filter 修复较大孔洞并保持边缘；
4. 输出修复前后有效率、RMSE 和图像。

实验元数据见 `data_card.yaml`，复现检查见 `CHECKLIST.md`。
实验参数固定在 `config.json`，固定评估划分说明见 `split_manifest.json`。

## 交互页面

双击 `run_ui.bat` 启动固定尺寸中俄双语 UI。页面使用 `baseball_sample.png` 作为棒球视觉引导图，可调节缺失率、孔洞大小和 Guided Filter 半径，点击“重新运行”查看四幅图和指标，点击“保存指标”写入 `results/interactive_metrics.json`。
