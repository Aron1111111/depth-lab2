# 试卷第二问代码 / Код для вторых вопросов

这三个脚本对应试卷 A、B、C 三部分的第二问。代码使用合成数据，直接运行即可看到结果，适合考试时书写和演示。
Эти три скрипта соответствуют вторым вопросам частей A, B и C. Используются синтетические данные, поэтому код можно запустить сразу.

```powershell
python A2_ransac_panorama.py
python B2_kalman_fusion.py
python C2_morphology.py
```

- `A2_ransac_panorama.py`：SIFT 特征 + 比值检验 + RANSAC 单应矩阵 + 全景拼接。
- `B2_kalman_fusion.py`：一维卡尔曼预测、两个传感器顺序更新和结果保存。
- `C2_morphology.py`：腐蚀、膨胀、开运算、闭运算和组合增强。

A2 和 C2 需要 OpenCV、NumPy；B2 只需要 NumPy。
A2 и C2 требуют OpenCV и NumPy; B2 требует только NumPy.
