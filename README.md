# Bike-Sharing-Demand-Prediction
## 项目描述

本项目使用TensorFlow 2.0构建了一个结合LSTM和多头注意力机制的深度学习模型，用于预测共享单车系统的使用需求。通过分析历史使用数据、环境因素和时间特征，模型实现了高精度的时间序列预测。

## 主要特点

- **创新的混合架构**: LSTM + 多头注意力机制 + 时间注意力
- **周期时间特征工程**: sin/cos编码处理小时和星期特征
- **动态学习率调整**: 指数衰减学习率策略
- **全面的模型诊断**: R²评分、残差分析、预测可视化
- **鲁棒的预处理**: RobustScaler归一化，时间序列滑动窗口

## 文件结构
bike-sharing-prediction/
├── data/
│ └── BikeShares.csv # 原始数据集
├── models/
│ └── best_model.h5 # 训练好的模型权重
├── logs/ # TensorBoard日志
├── 08_TensorFlow2.0_基于LSTM注意力机制实现多变量预测_共享单车使用量预测.py # 主代码文件
├──report/
    └── Project_Report.pdf    # 项目报告
└── README.md # 项目说明
依赖环境
Python 3.7+
TensorFlow 2.4+
ensorflow>=2.4.0
pandas>=1.1.5
numpy>=1.19.5
matplotlib>=3.3.4
seaborn>=0.11.1
scikit-learn>=0.24.1
jupyter>=1.0.0
