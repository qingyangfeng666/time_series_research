"""
ARIMA 信号检查版
只做一件事：生成信号表格，检查多空分布
不加任何阈值
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ===== 1. 加载数据 =====
file_path = "data/processed/ICL9_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

close = df['close'].values
returns = df['return'].values
dates = df['date'].values

# ===== 2. 切分 =====
returns_series = pd.Series(returns).dropna()
train_size = int(len(returns_series) * 0.8)
train, test = returns_series[:train_size], returns_series[train_size:]

print(f"训练集长度: {len(train)}, 测试集长度: {len(test)}")

# ===== 3. 训练ARIMA =====
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()

# ===== 4. 预测 =====
forecast = model_fit.forecast(steps=len(test))
forecast_values = forecast.values  # 转成numpy数组

# ===== 5. 检查预测值 =====
print("=" * 80)
print("ARIMA 预测值检查")
print("=" * 80)
print(f"预测值最大值: {forecast_values.max():.8f}")
print(f"预测值最小值: {forecast_values.min():.8f}")
print(f"预测值均值: {forecast_values.mean():.8f}")
print(f"预测值为正的数量: {(forecast_values > 0).sum()}")
print(f"预测值为负的数量: {(forecast_values < 0).sum()}")
print(f"预测值为零的数量: {(forecast_values == 0).sum()}")

# ===== 6. 生成信号 =====
signals = np.sign(forecast_values)

print("=" * 80)
print("信号分布")
print("=" * 80)
print(f"做多信号 (+1): {(signals == 1).sum()}")
print(f"做空信号 (-1): {(signals == -1).sum()}")
print(f"无信号 (0): {(signals == 0).sum()}")

# ===== 7. 输出前20行 =====
test_dates = dates[train_size + 1:train_size + 1 + len(test)]
test_close = close[train_size + 1:train_size + 1 + len(test)]

n = min(20, len(test_dates), len(forecast_values))
print("\n" + "=" * 80)
print("前20行数据预览")
print("=" * 80)
print(f"{'日期':<12} {'收盘价':>8} {'ARIMA预测值':>14} {'信号':>4}")
print("-" * 80)
for i in range(n):
    print(f"{str(test_dates[i]):<12} {test_close[i]:>8.2f} {forecast_values[i]:>14.8f} {signals[i]:>4}")