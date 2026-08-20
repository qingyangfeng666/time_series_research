"""
批量协整检验 - 找到有效配对
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("批量协整检验 - 寻找有效配对")
print("=" * 60)

# ===== 1. 加载数据 =====
file_path = "data/processed/ssdstock.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# 获取所有股票
stock_names = df['股票代码'].unique()
print(f"股票数量: {len(stock_names)}")

# ===== 2. 提取所有股票价格 =====
prices = {}
for stock in stock_names:
    tmp = df[df['股票代码'] == stock][['date', 'close']].set_index('date').sort_index()
    prices[stock] = tmp['close']

# ===== 3. 合并成一个DataFrame =====
price_df = pd.DataFrame(prices).dropna()

print(f"对齐后数据量: {len(price_df)} 行")
print(f"股票数量: {len(price_df.columns)}")

# ===== 4. 批量协整检验（只测前30只，节省时间） =====
stock_list = price_df.columns[:30]
results = []

print(f"\n正在检验前 {len(stock_list)} 只股票...")

for i in range(len(stock_list)):
    for j in range(i+1, len(stock_list)):
        s1, s2 = stock_list[i], stock_list[j]
        score, p_value, _ = coint(price_df[s1], price_df[s2])
        if p_value < 0.05:
            results.append({
                '股票1': s1,
                '股票2': s2,
                'p-value': p_value
            })

# ===== 5. 输出结果 =====
if len(results) == 0:
    print("\n❌ 前30只股票中没有找到协整关系")
else:
    df_results = pd.DataFrame(results).sort_values('p-value')
    print(f"\n✅ 找到 {len(results)} 对协整关系")
    print(df_results.to_string(index=False))

    # 显示最佳配对
    best = df_results.iloc[0]
    print(f"\n最佳配对: {best['股票1']} & {best['股票2']}")
    print(f"p-value: {best['p-value']:.4f}")