"""
VAR 向量自回归 - 独立模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("VAR 向量自回归")
print("=" * 60)

# ===== 1. 加载数据 =====
file_path = "data/processed/ssdstock.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# ===== 2. 提取两只股票（深桑达Ａ & 三花智控） =====
stock1, stock2 = '深桑达Ａ', '三花智控'
p1 = df[df['股票代码'] == stock1][['date', 'close']].set_index('date').sort_index()
p2 = df[df['股票代码'] == stock2][['date', 'close']].set_index('date').sort_index()

merged = pd.merge(p1, p2, left_index=True, right_index=True, suffixes=('_1', '_2')).dropna()
merged = merged / merged.iloc[0]  # 归一化

print(f"数据量: {len(merged)} 行")
print(f"两只股票: {stock1} & {stock2}")

# ===== 3. 计算收益率 =====
returns = merged.pct_change().dropna()
returns.columns = ['ret_1', 'ret_2']

print(f"收益率数据量: {len(returns)} 行")

# ===== 4. 平稳性检验（VAR要求数据平稳） =====
print("\n" + "=" * 60)
print("平稳性检验（ADF）")
print("=" * 60)

for col in returns.columns:
    result = adfuller(returns[col].dropna())
    print(f"{col}: p-value = {result[1]:.4f} {'✅ 平稳' if result[1] < 0.05 else '❌ 非平稳'}")

# ===== 5. 训练 VAR 模型 =====
print("\n" + "=" * 60)
print("训练 VAR 模型")
print("=" * 60)

# 切分训练/测试
train_size = int(len(returns) * 0.8)
train, test = returns[:train_size], returns[train_size:]

print(f"训练集: {len(train)} 行, 测试集: {len(test)} 行")

# 训练 VAR
model = VAR(train)
lag_order = model.select_order(maxlags=15)
print(f"\n最优滞后阶数 (AIC): {lag_order.aic}")

# 用 AIC 最优阶数训练
best_lag = lag_order.aic
model_fitted = model.fit(best_lag)
print(model_fitted.summary())

# ===== 6. 预测 =====
print("\n" + "=" * 60)
print("预测结果")
print("=" * 60)

forecast = model_fitted.forecast(train.values[-best_lag:], steps=len(test))
forecast_df = pd.DataFrame(forecast, index=test.index, columns=returns.columns)

# ===== 7. 计算预测准确率 =====
# 方向准确率：预测涨跌方向是否与实际一致
actual_direction = (test > 0).values
pred_direction = (forecast_df > 0).values

acc_1 = (actual_direction[:, 0] == pred_direction[:, 0]).mean()
acc_2 = (actual_direction[:, 1] == pred_direction[:, 1]).mean()

print(f"{stock1} 方向准确率: {acc_1:.2%}")
print(f"{stock2} 方向准确率: {acc_2:.2%}")

# ===== 8. 画图 =====
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 股票1
ax1 = axes[0]
ax1.plot(test.index, test['ret_1'], label='实际', color='blue', linewidth=0.8)
ax1.plot(test.index, forecast_df['ret_1'], label='预测', color='red', linestyle='--', linewidth=0.8)
ax1.set_title(f'{stock1} 收益率预测')
ax1.set_ylabel('收益率')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 股票2
ax2 = axes[1]
ax2.plot(test.index, test['ret_2'], label='实际', color='blue', linewidth=0.8)
ax2.plot(test.index, forecast_df['ret_2'], label='预测', color='red', linestyle='--', linewidth=0.8)
ax2.set_title(f'{stock2} 收益率预测')
ax2.set_xlabel('日期')
ax2.set_ylabel('收益率')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/var_model.png", dpi=150, bbox_inches='tight')
print(f"\n✅ 图片已保存: reports/var_model.png")
plt.show()