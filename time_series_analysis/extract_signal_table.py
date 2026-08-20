"""
提取单只股票的详细ARIMA信号表格
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "C:/Users/95722/projects/quant_research/data/processed/ssdstock.csv"
TARGET_STOCK = "兴齐眼药"

# ===== 1. 加载数据 =====
df = pd.read_csv(FILE_PATH, parse_dates=['date'])
df = df[['date', '股票代码', 'close']].copy()
df = df.rename(columns={'股票代码': 'stock'})

# ===== 2. 提取目标股票 =====
stock_df = df[df['stock'] == TARGET_STOCK][['date', 'close']].dropna()
close = stock_df['close'].values
dates = stock_df['date'].values

print(f"股票: {TARGET_STOCK}")
print(f"数据量: {len(close)} 行")

# ===== 3. 计算收益率并切分 =====
returns = np.diff(close) / close[:-1]
returns = np.append([0], returns)
returns_series = pd.Series(returns).dropna()
train_size = int(len(returns_series) * 0.8)
train, test = returns_series[:train_size], returns_series[train_size:]

# ===== 4. 训练ARIMA并预测 =====
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
forecast_values = forecast.values

# ===== 5. 生成信号 =====
signals = np.where(forecast_values > 0, 1, 0)

# ===== 6. 组装信号表格 =====
test_dates = dates[train_size + 1:train_size + 1 + len(test)]
test_close = close[train_size + 1:train_size + 1 + len(test)]
test_returns = returns[train_size + 1:train_size + 1 + len(test)]

n = min(len(signals), len(test_dates))
result_df = pd.DataFrame({
    '日期': test_dates[:n],
    '收盘价': test_close[:n],
    '收益率': test_returns[:n],
    '预测值': forecast_values[:n],
    '信号': signals[:n]
})
result_df['信号说明'] = result_df['信号'].map({1: '做多', 0: '空仓'})

# ===== 7. 计算策略收益 =====
result_df['策略收益率'] = result_df['信号'] * result_df['收益率']
result_df['累计策略收益'] = (1 + result_df['策略收益率']).cumprod() - 1
result_df['累计买入持有'] = (1 + result_df['收益率']).cumprod() - 1

# ===== 8. 计算详细指标 =====
strategy_returns = result_df['策略收益率'].dropna()
total_return = (1 + strategy_returns).prod() - 1
win_rate = (strategy_returns > 0).mean()
sharpe = strategy_returns.mean() / strategy_returns.std() * (252 ** 0.5) if strategy_returns.std() > 0 else 0

cumsum = np.cumsum(strategy_returns)
running_max = np.maximum.accumulate(cumsum)
drawdown = cumsum - running_max
max_drawdown = drawdown.min()

n_days = len(strategy_returns)
annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

print("\n" + "=" * 80)
print(f"{TARGET_STOCK} 策略指标")
print("=" * 80)
print(f"总收益率: {total_return:.2%}")
print(f"年化收益率: {annual_return:.2%}")
print(f"胜率: {win_rate:.2%}")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"交易次数: {(result_df['信号'].diff() != 0).sum()}")

# ===== 9. 保存信号表格 =====
output_path = f"C:/Users/95722/projects/quant_research/reports/{TARGET_STOCK}_signal_table.csv"
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 信号表格已保存: {output_path}")

# ===== 10. 显示前20行 =====
print("\n" + "=" * 80)
print(f"{TARGET_STOCK} 信号表格（前20行）")
print("=" * 80)
print(result_df[['日期', '收盘价', '预测值', '信号说明', '策略收益率', '累计策略收益']].head(20).to_string(index=False))

# ===== 11. 信号分布统计 =====
print("\n" + "=" * 80)
print("信号分布统计")
print("=" * 80)
print(result_df['信号说明'].value_counts())