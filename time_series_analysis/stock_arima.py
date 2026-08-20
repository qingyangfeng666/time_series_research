"""
导出开平仓明细表（次日开盘价开平仓）
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "C:/Users/95722/projects/quant_research/data/processed/ssdstock.csv"
TARGET_STOCK = "海星股"
TRAILING_PCT = 0.015

# ===== 1. 加载数据 =====
df = pd.read_csv(FILE_PATH, parse_dates=['date'])
df = df[['date', '股票代码', 'open', 'close']].copy()
df = df.rename(columns={'股票代码': 'stock'})

# ===== 2. 提取目标股票 =====
stock_df = df[df['stock'] == TARGET_STOCK][['date', 'open', 'close']].dropna()
open_price = stock_df['open'].values
close = stock_df['close'].values
dates = stock_df['date'].values

print(f"股票: {TARGET_STOCK}")
print(f"数据量: {len(close)} 行")

# ===== 3. 计算收益率 =====
returns = np.diff(close) / close[:-1]
returns = np.append([0], returns)
returns_series = pd.Series(returns).dropna()

# ===== 4. 滚动ARIMA预测 =====
window = 60
forecast_values = []

for i in range(len(returns_series)):
    train_window = returns_series.iloc[max(0, i-window):i].dropna()
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

forecast_values = np.array(forecast_values)

# ===== 5. 生成信号 =====
signals = np.where(forecast_values > 0, 1, 0)

# ===== 6. 回测（次日开盘价开平仓） =====
position = 0
entry_price = 0
entry_date = None
entry_idx = 0
peak_price = 0
trade_log = []

for i in range(60, len(signals) - 1):  # 留1天给次日开盘
    signal = signals[i]
    current_date = dates[i]

    if signal == 0:
        continue

    # ===== 开仓（信号次日开盘价） =====
    if position == 0:
        position = 1
        entry_price = open_price[i + 1]  # ✅ 次日开盘价
        entry_date = dates[i + 1]
        entry_idx = i + 1
        peak_price = entry_price
        continue

    # ===== 持仓中，追踪最高价 =====
    if close[i] > peak_price:
        peak_price = close[i]

    # 检查平仓条件
    should_close = False
    close_reason = ""

    # 回撤保护
    drawdown = (peak_price - close[i]) / peak_price
    if drawdown >= TRAILING_PCT:
        should_close = True
        close_reason = f"回撤保护（从最高{peak_price:.2f}回撤{drawdown:.2%}）"

    # 信号反转
    if signal == 0:
        should_close = True
        close_reason = "信号转空"

    if should_close:
        exit_price = open_price[i + 1]  # ✅ 平仓次日开盘价
        ret = exit_price / entry_price - 1
        trade_log.append({
            '开仓日期': entry_date,
            '平仓日期': dates[i + 1],
            '开仓价': entry_price,
            '平仓价': exit_price,
            '盈亏%': ret * 100,
            '持仓天数': (i + 1) - entry_idx,
            '平仓原因': close_reason,
            '开仓信号': '做多',
        })
        position = 0
        continue

# ===== 7. 输出 =====
if len(trade_log) == 0:
    print("没有交易记录")
else:
    df_trades = pd.DataFrame(trade_log)

    output_path = f"C:/Users/95722/projects/quant_research/reports/{TARGET_STOCK}_trades.xlsx"
    df_trades.to_excel(output_path, index=False, engine='openpyxl')

    print("\n" + "=" * 80)
    print(f"{TARGET_STOCK} 开平仓明细表（次日开盘价，共{len(df_trades)}笔交易）")
    print("=" * 80)
    print(df_trades.to_string(index=False))

    print(f"\n✅ 已保存: {output_path}")

    print("\n" + "=" * 80)
    print("交易统计")
    print("=" * 80)
    print(f"总交易次数: {len(df_trades)}")
    print(f"盈利次数: {(df_trades['盈亏%'] > 0).sum()}")
    print(f"亏损次数: {(df_trades['盈亏%'] < 0).sum()}")
    print(f"胜率: {(df_trades['盈亏%'] > 0).mean():.2%}")
    print(f"总收益率: {df_trades['盈亏%'].sum():.2f}%")
    print(f"最大单笔盈利: {df_trades['盈亏%'].max():.2f}%")
    print(f"最大单笔亏损: {df_trades['盈亏%'].min():.2f}%")
    print(f"平均持仓天数: {df_trades['持仓天数'].mean():.1f}天")
    print("\n平仓原因分布:")
    print(df_trades['平仓原因'].value_counts())