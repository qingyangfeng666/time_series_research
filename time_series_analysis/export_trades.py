"""
导出开平仓明细表（盘中回撤判断 + 次日开盘价开平仓）
TRAILING_PCT 可调，1.5% 对应当前结果
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ===== 用户可调参数 =====
FILE_PATH = "C:/Users/95722/projects/quant_research/data/processed/ssdstock.csv"
TARGET_STOCK = "海星股"
TRAILING_PCT = 0.05  # 回撤保护阈值，可以改成 0.02、0.03 等
WINDOW = 60            # 滚动窗口天数

# ===== 1. 加载数据 =====
df = pd.read_csv(FILE_PATH, parse_dates=['date'])
df = df[['date', '股票代码', 'open', 'high', 'low', 'close']].copy()
df = df.rename(columns={'股票代码': 'stock'})

# ===== 2. 提取目标股票 =====
stock_df = df[df['stock'] == TARGET_STOCK][['date', 'open', 'high', 'low', 'close']].dropna()
open_price = stock_df['open'].values
high = stock_df['high'].values
low = stock_df['low'].values
close = stock_df['close'].values
dates = stock_df['date'].values

print(f"股票: {TARGET_STOCK}")
print(f"数据量: {len(close)} 行")
print(f"回撤保护阈值: {TRAILING_PCT*100:.1f}%")

# ===== 3. 计算收益率 =====
returns = np.diff(close) / close[:-1]
returns = np.append([0], returns)
returns_series = pd.Series(returns).dropna()

# ===== 4. 滚动ARIMA预测 =====
forecast_values = []
total = len(returns_series)

print("滚动ARIMA预测中...")
for i in range(total):
    if i % 50 == 0:
        print(f"  进度: {i}/{total}")
    train_window = returns_series.iloc[max(0, i-WINDOW):i].dropna()
    if len(train_window) < 30:
        forecast_values.append(0)
        continue
    try:
        model = ARIMA(train_window, order=(1, 0, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        forecast_values.append(forecast.iloc[0])
    except Exception:
        forecast_values.append(0)
print(f"  进度: {total}/{total} 完成")

forecast_values = np.array(forecast_values)

# ===== 5. 生成信号 =====
signals = np.where(forecast_values > 0, 1, 0)

# ===== 6. 回测（盘中回撤判断 + 次日开盘价开平仓） =====
position = 0
entry_price = 0
entry_date = None
entry_idx = 0
peak_price = 0
trade_log = []

print("回测中（盘中回撤判断）...")
for i in range(WINDOW, len(signals) - 1):
    signal = signals[i]

    if signal == 0:
        continue

    if position == 0:
        position = 1
        entry_price = open_price[i + 1]
        entry_date = dates[i + 1]
        entry_idx = i + 1
        peak_price = entry_price
        continue

    # ✅ 盘中回撤判断：用当日最高价追踪，用当日最低价判断
    if high[i] > peak_price:
        peak_price = high[i]

    should_close = False
    close_reason = ""

    # 盘中回撤保护：用最低价（代表盘中最大回撤）判断
    drawdown = (peak_price - low[i]) / peak_price
    if drawdown >= TRAILING_PCT:
        should_close = True
        close_reason = f"盘中回撤（最低{low[i]:.2f}，回撤{drawdown:.2%}）"

    if signal == 0:
        should_close = True
        close_reason = "信号转空"

    if should_close:
        # exit_price = open_price[i + 1]  # 次日开盘价平仓
        exit_price = peak_price * (1 - TRAILING_PCT)  # 理论回撤价位
        ret = exit_price / entry_price - 1
        trade_log.append({
            '开仓日期': entry_date,
            '平仓日期': dates[i + 1],
            '开仓价': entry_price,
            '平仓价': exit_price,
            '盈亏%': ret * 100,
            '持仓天数': (i + 1) - entry_idx,
            '平仓原因': close_reason,
        })
        position = 0
        continue

# ===== 7. 输出 =====

if len(trade_log) == 0:
    print("没有交易记录")
else:
    df_trades = pd.DataFrame(trade_log)

    # 计算总收益率
    total_return = df_trades['盈亏%'].sum() / 100

    # 计算交易天数（从第一笔开仓到最后一笔平仓）
    first_open = df_trades['开仓日期'].min()
    last_close = df_trades['平仓日期'].max()
    trading_days = (last_close - first_open).days
    years = trading_days / 365

    # 年化收益率
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 夏普比率（假设无风险利率为0）
    returns_series = df_trades['盈亏%'] / 100
    sharpe = returns_series.mean() / returns_series.std() * (252 ** 0.5) if returns_series.std() > 0 else 0

    print("\n" + "=" * 80)
    print("交易统计")
    print("=" * 80)
    print(f"总交易次数: {len(df_trades)}")
    print(f"盈利次数: {(df_trades['盈亏%'] > 0).sum()}")
    print(f"亏损次数: {(df_trades['盈亏%'] < 0).sum()}")
    print(f"胜率: {(df_trades['盈亏%'] > 0).mean():.2%}")
    print(f"总收益率: {df_trades['盈亏%'].sum():.2f}%")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大单笔盈利: {df_trades['盈亏%'].max():.2f}%")
    print(f"最大单笔亏损: {df_trades['盈亏%'].min():.2f}%")
    print(f"平均持仓天数: {df_trades['持仓天数'].mean():.1f}天")
    print("\n平仓原因分布:")
    print(df_trades['平仓原因'].value_counts())