"""
独立时间序列分析项目
只读数据，不改任何原有文件
输出：ADF检验结果 + ACF/PACF图 + ARIMA预测图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# 在 plt 已经导入之后，加上这两行
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
print("=" * 60)
print("时间序列分析：中证500股指期货")
print("=" * 60)

# ===== 1. 加载数据（和量化项目共用） =====
file_path = "data/processed/ICL9_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['date'])
print(f"数据量: {len(df)} 行")

# ===== 2. 提取收盘价 =====
close = df['close'].dropna()
print(f"收盘价序列长度: {len(close)}")

# ===== 3. 原始价格图 =====
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['close'])
plt.title('中证500股指期货原始收盘价')
plt.xlabel('日期')
plt.ylabel('价格')
plt.grid(True)
plt.savefig('time_series_analysis/original_price.png', dpi=150)
print("✅ 原始价格图已保存: time_series_analysis/original_price.png")

# ===== 4. ADF检验（原始序列） =====
print("\n" + "=" * 60)
print("ADF 平稳性检验（原始序列）")
print("=" * 60)

result = adfuller(close, autolag='AIC')
print(f"ADF 统计量: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print(f"临界值:")
for key, value in result[4].items():
    print(f"  {key}: {value:.4f}")

if result[1] < 0.05:
    print("✅ 结论: p-value < 0.05，序列平稳")
else:
    print("⚠️ 结论: p-value >= 0.05，序列非平稳（取收益率）")

# ===== 5. 计算收益率 =====
returns = close.pct_change().dropna()
print(f"\n收益率序列长度: {len(returns)}")

# ===== 6. ADF检验（收益率） =====
print("\n" + "=" * 60)
print("ADF 平稳性检验（收益率序列）")
print("=" * 60)

result_ret = adfuller(returns, autolag='AIC')
print(f"ADF 统计量: {result_ret[0]:.4f}")
print(f"p-value: {result_ret[1]:.4f}")
for key, value in result_ret[4].items():
    print(f"  {key}: {value:.4f}")

if result_ret[1] < 0.05:
    print("✅ 收益率序列平稳，适合建模")

# ===== 7. ACF/PACF图 =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(returns, lags=20, ax=ax1)
plot_pacf(returns, lags=20, ax=ax2)
ax1.set_title('收益率自相关 (ACF)')
ax2.set_title('收益率偏自相关 (PACF)')
plt.tight_layout()
plt.savefig('time_series_analysis/acf_pacf.png', dpi=150)
print("✅ ACF/PACF 图已保存: time_series_analysis/acf_pacf.png")

# ===== 8. ARIMA模型 =====
print("\n" + "=" * 60)
print("ARIMA 模型预测")
print("=" * 60)

train_size = int(len(returns) * 0.8)
train, test = returns[:train_size], returns[train_size:]

model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=len(test))
mse = ((forecast - test) ** 2).mean()
print(f"\n均方误差 (MSE): {mse:.6f}")

# ===== 9. 预测 vs 实际 =====
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='实际收益率', color='blue')
plt.plot(test.index, forecast, label='ARIMA预测', color='red', linestyle='--')
plt.title('ARIMA(1,0,1) 预测 vs 实际')
plt.xlabel('日期')
plt.ylabel('收益率')
plt.legend()
plt.grid(True)
plt.savefig('time_series_analysis/arima_forecast.png', dpi=150)
print("✅ 预测对比图已保存: time_series_analysis/arima_forecast.png")

print("\n" + "=" * 60)
print("✅ 时间序列分析完成！")
print("   生成文件:")
print("   - time_series_analysis/original_price.png")
print("   - time_series_analysis/acf_pacf.png")
print("   - time_series_analysis/arima_forecast.png")
print("=" * 60)