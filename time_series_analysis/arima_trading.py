"""
检查ARIMA是否用了未来数据
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

file_path = "data/processed/ICL9_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

close = df['close'].values
returns = df['return'].values
dates = df['date'].values

# 切分
returns_series = pd.Series(returns).dropna()
train_size = int(len(returns_series) * 0.8)
train = returns_series[:train_size]
test = returns_series[train_size:]

print(f"训练集: {train.index[0]} 到 {train.index[-1]}")
print(f"测试集: {test.index[0]} 到 {test.index[-1]}")

# 训练ARIMA
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()

# 预测测试集
forecast = model_fit.forecast(steps=len(test))

print("=" * 80)
print("2026年7月预测值检查")
print("=" * 80)

test_dates = dates[train_size + 1:train_size + 1 + len(test)]
for i, d in enumerate(test_dates):
    if pd.Timestamp(d) >= pd.Timestamp('2026-07-01') and pd.Timestamp(d) <= pd.Timestamp('2026-07-31'):
        print(f"{d} 预测值: {forecast.iloc[i]:.6f}")