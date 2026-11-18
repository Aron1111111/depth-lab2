# 多传感器同步与标定实验

这是课程“同步和标定多传感器数据”的完整实验项目。程序使用固定随机种子生成可重复的 Camera、LiDAR、IMU 模拟数据，并提供真实摄像头交互界面，不要求实体 LiDAR 或 IMU 硬件。

## 运行

双击 `run.bat`，或执行：

```powershell
D:\002_codex\image_class\.venv\Scripts\python.exe calibration_lab.py
```

程序会自动生成：

- `results/calib_v1.yaml`：时间、内参、外参和指标；
- `results/metrics.json`：机器可读实验指标；
- `results/sync_timeline.png`：时间轴同步图；
- `results/reprojection_error.png`：标定前后投影对比；
- `REPORT.md`：实验报告；
- `commit_hash.txt`：当前代码版本信息。

## 交互界面

界面为中俄双语，包含：

- 电脑自带摄像头实时画面；
- 没有摄像头时自动使用演示画面；
- LiDAR/Radar 模拟点云；
- IMU 模拟角速度和噪声；
- 时间偏移 `dt` 滑块；
- LiDAR 到 Camera 的 yaw、tx、ty 滑块；
- LiDAR 距离、噪声和 IMU 噪声滑块；
- 同步通过率、匹配率和重投影误差的实时显示；
- “保存参数 / Сохранить”按钮会生成 `results/interactive_config.json`。

滑块故意改变了同步和标定状态：例如把 `dt` 从 50 ms 拖远，匹配率会下降；把 yaw 或平移拖离标定值，重投影误差会升高。

## 完成的知识点

- 时间偏移 `dt` 估计；
- 线性插值和不同采样率匹配；
- jitter 和 match rate；
- Kabsch/SVD 外参估计；
- 相机内参矩阵 `K`；
- LiDAR 到相机的旋转 `R` 与平移 `t`；
- 重投影误差；
- 标定结果的验收指标。

## 交付清单

实验提交包括代码、`calib_v1.yaml`、`metrics.json`、报告、README、图表和版本号。报告中还解释了同步与标定的区别、最近邻误差、重投影误差、立体校正和 `dt` 评价方法。
