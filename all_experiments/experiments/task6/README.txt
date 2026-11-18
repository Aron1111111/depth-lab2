猫狗分类实验

启动
双击 run.bat。界面默认载入上次训练结果。
三个页面依次显示数据概览、训练与验证、测试评价。
修改参数后点击“训练并比较两模型”。
“训练狗保留比例”用于模拟仅在训练集减少狗样本，验证和测试分布保持自然。
点击“识别照片”可查看 SVM 与随机森林对所选照片的分类。

数据来源
TensorFlow 官方教学数据 Cats and Dogs Filtered，原始猫狗照片来自 Dogs vs. Cats。
https://www.tensorflow.org/tutorials/images/transfer_learning
教学下载链接
https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip
本次实际下载链接
https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip
原压缩包保存在 dataset/cats_and_dogs_filtered.zip。
图片在 dataset/cats_and_dogs_filtered/train 和 validation 下的 cats、dogs 文件夹。
保留源数据目录，实验读取后按固定随机种子重新分层划分训练/验证/测试。
加载时检查解码像素与图片尺寸的 SHA256，过滤完全相同的图片。
划分清单保存在 results/split_manifest.json。
照片原始尺寸不同，特征提取前统一成 64×64 灰度图。
HOG 为 8×8 像素单元、2×2 单元块、9 个方向，描述符长 1764。
标准化在 SVM 的 Pipeline 内拟合，交叉验证不会先用全部数据拟合标准化。
类别指标采用宏平均，ROC-AUC 以“狗”为正类。
SVM 使用 decision_function 分数，随机森林使用狗类别概率计算 ROC-AUC。
这是课程要求的手工特征与传统分类器实验，性能取决于当前特征与参数。

依赖
numpy、Pillow、scikit-learn、joblib
