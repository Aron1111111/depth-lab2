# 实验二结果报告

## 方法

使用固定随机种子 `20241110` 生成 260×180 的合成深度图，随机删除约 10% 深度值，并加入两个连续缺失区域。随后执行有效性掩码、3×3 形态学闭运算和带掩码 Guided Filter。

## 结果

运行 `reproduce_all.bat` 后，指标保存在 `results/metrics_before_after.json`。本次运行结果：

- 修复前有效像素：86.71%
- 修复后有效像素：96.36%
- 缺失区域 RMSE：0.3372 m
- 全图 RMSE：0.1229 m

输出图像：`depth_ground_truth.png`、`depth_corrupted.png`、`depth_restored.png`、`validity_mask.png`。
