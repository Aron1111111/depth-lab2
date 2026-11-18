猫狗 CNN 课程实验

本次最后一张课件的题目
“根据自己的任务选择架构，并说明理由”。

本项目的回答
任务是对整张照片进行猫或狗分类，因此选择 CNN。
卷积层从图像中学习局部特征，ReLU 提供非线性，最大池化降低特征图空间尺寸。
后续卷积组合更复杂的特征，全连接层输出猫/狗类别。
随机水平翻转、小角度旋转、平移、缩放仅用于训练集。
以验证集损失选择最佳轮次，最终评价使用固定测试集。

课件中的其他架构按任务选择
需要输出物体类别和位置时，可选 YOLO 或 Faster R-CNN。
需要逐像素语义类别时，可选 U-Net。
需要区分同类别的不同对象并输出各自掩膜时，可选 Mask R-CNN。
速度、精度取决于具体模型、输入大小、硬件及数据，需要实际比较。

使用方式
双击 run.bat 启动中俄双语界面。
默认载入已训练模型，显示输入照片、第一层卷积和池化特征图。
可拖动特征通道滑动器查看 16 个第一层通道。
可调整轮次和学习率、开关数据增强，点击“训练 CNN”重新训练。
“打开照片并识别”展示猫/狗预测、softmax 得分和单次推理耗时。
softmax 得分是模型输出，尚未做概率校准。
训练过程页展示训练/验证损失和准确率。
识别与评价页展示固定测试集指标、CNN 混淆矩阵及 task6 模型对照。
Precision、Recall、F1 为猫狗两类宏平均；ROC-AUC 以狗为正类。

数据
沿用 task6/dataset/cats_and_dogs_filtered，未重复下载。
任务开始时的 task6 划分和结果分别复制到 split_manifest.json、baseline_metrics.json。
后续修改 task6 的划分不影响本次任务已固定的划分。
图片按路径从 task6/dataset 读取，删除该数据集后需要恢复才能使用。
manifest 中 train、validation、test 路径无交集。
训练采用 64×64 RGB，批大小 64，Adam，默认 10 轮。
CNN 从随机初始化训练，权重保存到 results/cnn.pt。
CPU 训练耗时取决于轮次和电脑配置。
经典 HOG 模型与 CNN 使用不同特征，表格比较的是完整方法在同一测试集上的结果。

运行环境
D:/Anaconda/Anaconda/envs/yolov8_env/python.exe
该环境已有 torch、torchvision、Pillow、NumPy。
如果从 PyCharm 运行 cnn_lab.py，请为该运行配置选择上述解释器。

命令行重新训练
python cnn_lab.py --train --epochs 10
