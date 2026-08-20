"""
GARCH 波动率建模 - 独立模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("GARCH 波动率建模")
print("=" * 60)

# ===== 1. 加载数据 =====
file_path = "data/processed/ICL9_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['date'])
returns = df['return'].dropna() * 100  # 转为百分比

print(f"数据量: {len(returns)} 行")

# ===== 2. 划分训练/测试 =====
train_size = int(len(returns) * 0.8)
train, test = returns[:train_size], returns[train_size:]

print(f"训练集: {len(train)} 行, 测试集: {len(test)} 行")

# ===== 3. 训练 GARCH(1,1) 模型 =====
print("\n训练 GARCH(1,1) 模型...")
model = arch_model(train, vol='Garch', p=1, q=1)
res = model.fit(disp='off')
print(res.summary())

# ===== 4. 预测波动率 =====
forecast = res.forecast(horizon=len(test))
predicted_vol = np.sqrt(forecast.variance.values[-1, :])

# ===== 5. 计算实际波动率（滚动20天） =====
actual_vol = test.rolling(20).std()

# ===== 6. 对比预测 vs 实际 =====
print("\n" + "=" * 60)
print("波动率预测结果")
print("=" * 60)
print(f"预测波动率均值: {predicted_vol.mean():.4f}")
print(f"实际波动率均值: {actual_vol.mean():.4f}")

# ===== 7. 画图 =====
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 上：收益率序列
ax1 = axes[0]
ax1.plot(df['date'], returns, color='blue', linewidth=0.8)
ax1.set_title('中证500 日收益率')
ax1.set_xlabel('日期')
ax1.set_ylabel('收益率 (%)')
ax1.grid(True, alpha=0.3)

# 下：波动率预测 vs 实际
ax2 = axes[1]
ax2.plot(test.index, predicted_vol, color='red', linewidth=1.5, label='GARCH预测波动率')
ax2.plot(test.index, actual_vol, color='blue', linewidth=0.8, alpha=0.7, label='实际波动率(滚动20天)')
ax2.set_title('GARCH 波动率预测 vs 实际')
ax2.set_xlabel('日期')
ax2.set_ylabel('波动率 (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/garch_volatility.png", dpi=150, bbox_inches='tight')
print(f"\n✅ 图片已保存: reports/garch_volatility.png")
plt.show()