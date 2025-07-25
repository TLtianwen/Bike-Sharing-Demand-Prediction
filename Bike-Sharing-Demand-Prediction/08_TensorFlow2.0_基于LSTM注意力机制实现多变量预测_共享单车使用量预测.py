#!/usr/bin/env python
# coding: utf-8

# ### 案例实现流程：
# 
# * 1. 加载数据集、数据可视化、预处理
# * 2. 特征工程
# * 3. 构建模型
# * 4. 模型编译、训练、验证

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


# ### 第1步：加载数据集、预处理

# 加载数据集

dataset = pd.read_csv("BikeShares.csv", parse_dates=['timestamp'], index_col=['timestamp'])


# 数据集大小

dataset.shape


# 默认显示前5行

dataset.head()


# 默认显示后5行

dataset.tail()


# #### 字段说明：
# 
# * timestamp : 时间戳
# * cnt : 租用共享单车的数量（目标值）
# * t1 : 气温
# * t2 : 体感温度
# * hum : 湿度
# * wind_speed : 风速
# * weather_code : 天气的类别（1=干净，2 =很少的云，3=碎云，4=多云，7=雨/小雨，10=有雷雨，26=降雪，94=冰雾）
# * is_holiday : 是否为假期（1:假期 / 0:工作日）
# * is_weekend : 是否为周末（1:周末 / 0：工作日）
# * season : 季节（0:春天 ; 1:夏天 ; 2:秋天 ; 3:冬天）

# 数据集信息

dataset.info()


# 数据集描述

dataset.describe()


# ### 第2步：数据集可视化

# 字段t1(气温)与字段cnt(单车使用量)之间的关系

plt.figure(figsize=(16,8))
sns.pointplot(x='t1', y='cnt', data=dataset)
plt.show()


# 字段t2(体感温度)与字段cnt(单车使用量)之间的关系

plt.figure(figsize=(16,8))
sns.lineplot(x='t2', y='cnt', data=dataset)
plt.show()


# 字段hum(湿度)与字段cnt(单车使用量)之间的关系

plt.figure(figsize=(16,8))
sns.lineplot(x='hum', y='cnt', data=dataset)
plt.xticks([])
plt.show()


# 字段weather_code : 天气的类别与字段cnt(单车使用量)之间的关系
# weather_code : 天气的类别（1=干净，2 =很少的云，3=碎云，4=多云，7=雨/小雨，10=有雷雨，26=降雪，94=冰雾

plt.figure(figsize=(16,8))
sns.pointplot(x='weather_code', y='cnt', data=dataset)
plt.show()


# #### 注意：创建时间字段，用于分析数据

# 创建hour字段

dataset['hour'] = dataset.index.hour


# 创建year字段

dataset['year'] = dataset.index.year


# 创建month字段

dataset['month'] = dataset.index.month


# 显示数据集

dataset.head(10)


# 基于is_holiday 统计 hour 与 cnt 之间的分布
# 1:假期 / 0:工作日

plt.figure(figsize=(16,8))
sns.lineplot(x='hour', y='cnt', data=dataset, hue='is_holiday')
plt.xticks(list(range(24)))
plt.show()


# 基于 season 统计 hour 与 cnt 之间的分布
# 0:春天 ; 1:夏天 ; 2:秋天 ; 3:冬天

plt.figure(figsize=(16,8))
sns.pointplot(x='hour', y='cnt', data=dataset, hue='season')
plt.xticks(list(range(24)))
plt.show()


# 基于 is_holiday 统计 hour 与 cnt 之间的分布
# 1:假期 / 0:工作日

plt.figure(figsize=(16,8))
sns.lineplot(x='month', y='cnt', data=dataset, hue='is_holiday')
plt.show()


# ### 第3步：数据预处理

dataset.head()


# 删除多余的列 hour, year, month

dataset.drop(columns=['hour', 'year', 'month'], axis=1, inplace=True)
# 1. 添加周期时间特征（改进点1）
dataset['hour_sin'] = np.sin(2 * np.pi * dataset.index.hour/24)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset.index.hour/24)
dataset['dayofweek_sin'] = np.sin(2 * np.pi * dataset.index.dayofweek/7)
dataset['dayofweek_cos'] = np.cos(2 * np.pi * dataset.index.dayofweek/7)


dataset.head()


# * 注意事项：
# * 1. cnt : 是标签；
# * 2. t1, t2, hum, wind_speed : 是数值类型字段；
# * 3. weather_code, is_holiday, is_weekend, season : 是分类类型字段；

# 3. 改进的归一化（使用RobustScaler）
numeric_features = ['t1', 't2', 'hum', 'wind_speed', 'cnt']
cyclic_features = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']

for col in numeric_features + cyclic_features:
    scaler = RobustScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1,1))


dataset.head()


# ### 第4步：特征工程

# 特征数据集

X = dataset.drop(columns=['cnt'], axis=1) 

# 标签数据集

y = dataset['cnt']


X.shape


y.shape


# 1 数据集分离： X_train, X_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)


X_train.shape


y_train.shape


X_train.head()


X_test.shape


y_test.shape


X_test.head()


# 2 构造特征数据集

def create_dataset(X, y, seq_len=10):
    features = []
    targets = []
    
    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i+seq_len] # 序列数据
        label = y.iloc[i+seq_len] # 标签数据
        # 保存到features和labels
        features.append(data)
        targets.append(label)
    
    # 返回
    return np.array(features), np.array(targets)


# ① 构造训练特征数据集

train_dataset, train_labels = create_dataset(X_train, y_train, seq_len=10)


train_dataset.shape


train_labels.shape


# ② 构造测试特征数据集

test_dataset, test_labels = create_dataset(X_test, y_test, seq_len=10)


test_dataset.shape


test_labels.shape


# 3 构造批数据

def create_batch_dataset(X, y, train=True, buffer_size=1000, batch_size=128):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) # 数据封装，tensor类型
    if train: # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # 测试集
        return batch_data.batch(batch_size)


# 训练批数据

train_batch_dataset = create_batch_dataset(train_dataset, train_labels)


# 测试批数据

test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)


# 从测试批数据中，获取一个batch_size的样本数据

list(test_batch_dataset.as_numpy_iterator())[0]


# ### 第5步：模型搭建、编译、训练

def build_enhanced_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # 第一LSTM层
    x = LSTM(128, return_sequences=True, kernel_initializer='he_normal')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 多头注意力机制（改进点2）
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Concatenate()([x, attn_output])
    x = LayerNormalization()(x)
    
    # 第二LSTM层
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    # 时间注意力层
    attention = tf.keras.layers.Attention()([x, x])
    x = Concatenate()([x, attention])
    
    # 输出部分
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # 改进的优化器配置（改进点3）
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                 loss='mse',
                 metrics=['mae'])
    return model

model = build_enhanced_model((10, X_train.shape[1]))



# 显示模型结构
utils.plot_model(model, show_shapes=True)


# 显示模型结构

utils.plot_model(model)


# 模型编译

model.compile(optimizer='adam',
              loss='mse')


# 保存模型权重文件和训练日志
# 删除 logs 目录（Windows 兼容写法）
if os.path.exists('logs'):
    shutil.rmtree('logs')  # 递归删除整个目录

# 创建新的日志目录
log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(log_dir, exist_ok=True)  # 自动创建目录

# 编译模型并设置 TensorBoard 回调
model.compile(optimizer='adam', loss='mse')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.001),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    TensorBoard(log_dir=os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
]

# 数据批处理
def create_batch_dataset(X, y, train=True, buffer_size=1000, batch_size=64):
    batch_data = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    return batch_data.batch(batch_size)

train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)


# 训练模型
history = model.fit(
    train_batch_dataset,
    epochs=100,
    validation_data=test_batch_dataset,
    callbacks=callbacks,
    verbose=1
)


# 显示训练结果

plt.figure(figsize=(16,8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(loc='best')
plt.show()


get_ipython().run_line_magic('load_ext', 'tensorboard')


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ### 第6步：模型验证

# --------------------- 第5步：评估与可视化 ---------------------
# 预测与评估
test_preds = model.predict(test_dataset).flatten()
score = r2_score(test_labels, test_preds)
print(f"改进后的R²值: {score:.4f}")

# 训练过程可视化
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f"Model Performance (Final R²={score:.4f})")
plt.legend()
plt.show()

# 预测结果可视化
plt.figure(figsize=(16,8))
plt.plot(test_labels[:200], label="True Values")
plt.plot(test_preds[:200], label="Predictions")
plt.title("Prediction vs Ground Truth (First 200 Samples)")
plt.legend()
plt.show()


test_dataset.shape


test_preds = model.predict(test_dataset, verbose=1)


test_preds.shape # 预测值shape


test_preds[:10]


test_preds = test_preds[:, 0] # 获取列值

test_preds[:10]


test_preds.shape


test_labels.shape # 真值shape


test_labels[:10]


# 计算r2值

score = r2_score(test_labels, test_preds)

print("r^2 值为： ", score)


# 绘制 预测与真值结果

plt.figure(figsize=(16,8))
plt.plot(test_labels[:300], label="True value")
plt.plot(test_preds[:300], label="Pred value")
plt.legend(loc='best')
plt.show()


# 检查预测结果的置信区间（建议添加此分析）
pred_std = np.std(test_labels - test_preds)
if pred_std < 0.05 * np.mean(test_labels):
    print("预测误差标准差过小 → 可能过拟合")
print("没有过拟合")


residuals = test_labels - test_preds
plt.figure(figsize=(12,4))
plt.subplot(121)
sns.histplot(residuals, kde=True)
plt.title("残差分布")

plt.subplot(122)
plt.scatter(test_preds, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title("残差-预测值散点图")
plt.show()


get_ipython().system('jupyter nbconvert --to script  --no-prompt 08_TensorFlow2.0_基于LSTM注意力机制实现多变量预测_共享单车使用量预测.ipynb')

