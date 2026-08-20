"""
LSTM 时序预测 - 独立模块
用过去60天收盘价预测未来5天走势
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("LSTM 时序预测")
print("=" * 60)

# ===== 1. 加载数据 =====
file_path = "data/processed/ICL9_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['date'])
close = df['close'].values

print(f"数据量: {len(close)} 行")

# ===== 2. 数据标准化 =====
scaler = MinMaxScaler()
close_scaled = scaler.fit_transform(close.reshape(-1, 1)).flatten()

# ===== 3. 创建序列数据 =====
def create_sequences(data, seq_length=60, pred_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
PRED_LENGTH = 5

X, y = create_sequences(close_scaled, SEQ_LENGTH, PRED_LENGTH)
print(f"样本数: {len(X)}")
print(f"X 形状: {X.shape}, y 形状: {y.shape}")

# ===== 4. 训练/测试切分 =====
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

# ===== 5. 构建 LSTM 模型 =====
print("\n构建 LSTM 模型...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(PRED_LENGTH)
])

model.compile(optimizer='adam', loss='mse')
print(model.summary())

# ===== 6. 训练 =====
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n训练 LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ===== 7. 预测 =====
y_pred = model.predict(X_test)

# 还原到原始价格
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, PRED_LENGTH)
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, PRED_LENGTH)

# ===== 8. 计算预测误差 =====
mse = np.mean((y_pred_original - y_test_original) ** 2)
mae = np.mean(np.abs(y_pred_original - y_test_original))

print(f"\n测试集误差:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# ===== 9. 画图 =====
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 训练损失
ax1 = axes[0]
ax1.plot(history.history['loss'], label='训练损失')
ax1.plot(history.history['val_loss'], label='验证损失')
ax1.set_title('LSTM 训练损失')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 预测 vs 实际（只画第一个预测日的对比）
ax2 = axes[1]
ax2.plot(y_test_original[:, 0], label='实际', color='blue', linewidth=0.8)
ax2.plot(y_pred_original[:, 0], label='预测', color='red', linestyle='--', linewidth=0.8)
ax2.set_title('LSTM 预测 vs 实际（第1天）')
ax2.set_xlabel('测试样本')
ax2.set_ylabel('价格')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/lstm_forecast.png", dpi=150, bbox_inches='tight')
print(f"\n✅ 图片已保存: reports/lstm_forecast.png")
plt.show()