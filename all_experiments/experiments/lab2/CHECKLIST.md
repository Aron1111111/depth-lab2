# Чек-лист воспроизводимости / 可复现性检查清单

- [x] 代码版本固定（Git commit 见仓库日志）
- [x] 依赖固定（`requirements.txt` 使用精确版本）
- [x] 环境可隔离（venv）
- [x] Random seed 固定为 `20241110`
- [x] 配置和元数据写入 `config.json`、`data_card.yaml`
- [x] 数据生成规则固定（`config.json`）
- [x] 固定评估划分已记录（`split_manifest.json`）
- [x] 数据/标定版本写入 Data Card
- [x] 指标记录到 `results/metrics_before_after.json`
- [x] 提供 `reproduce_all.bat`
- [x] 保存深度图、掩码和指标等结果产物
