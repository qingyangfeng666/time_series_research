# 时间序列分析：中证500与个股择时策略

> 📌 **与主项目的关系**：本仓库是独立的**时间序列研究模块**，与主仓库 `quant_research_project`（CTA策略）互为补充。主仓库负责策略构建与回测，本仓库负责时序模型的研究与验证。

## 项目简介
本项目系统性地探索了ARIMA、GARCH、协整检验、VAR和LSTM等时间序列模型在金融数据中的应用，覆盖了从平稳性检验、波动率建模到配对交易和深度学习预测的完整流程。
# 时间序列分析：中证500与个股择时策略

## 项目简介
本项目系统性地探索了ARIMA、GARCH、协整检验、VAR和LSTM等时间序列模型在金融数据中的应用，覆盖了从平稳性检验、波动率建模到配对交易和深度学习预测的完整流程。

## 五个模块

### 1. ADF检验 + ACF/PACF + ARIMA预测
- ADF检验：原始价格非平稳（p=0.32），收益率平稳（p=0.00）
- ACF/PACF：收益率存在一阶自相关
- ARIMA(1,0,1)：预测值在测试期内恒为正，无法提供有效择时信号

### 2. ARIMA择时策略 + 回撤保护
- 滚动60天窗口训练，预测次日收益率方向
- 回撤保护阈值3%，最大单笔亏损控制在-3.5%以内
- 测试期总收益率126.31%，年化72.98%，夏普5.07（理论回测值）

### 3. GARCH波动率建模
- GARCH(1,1) 模型：alpha=0.11，beta=0.87，alpha+beta≈0.98
- 波动率冲击具有高度持续性，适合动态仓位管理

### 4. 协整检验（配对交易）
- 在122只股票中批量检验，找到53对协整关系
- 最佳配对：深桑达Ａ & 三花智控，p-value=0.00004

### 5. VAR向量自回归
- 对深桑达Ａ和三花智控的收益率序列建模
- AIC最优滞后阶数为0，两股票间无互相预测关系

### 6. LSTM时序预测
- 输入过去60天价格，预测未来5天
- MAE约194元，预测精度有限

## 项目结构
```
time_series_analysis/
├── arima_analysis.py          # ADF检验 + ACF/PACF + ARIMA预测
├── arima_trading.py           # ARIMA择时策略回测
├── multi_stock_arima.py       # 多股票批量回测
├── export_trades.py           # 导出开平仓明细表
├── garch_volatility.py        # GARCH波动率建模
├── cointegration.py           # 协整检验 + 配对交易
├── var_model.py               # VAR向量自回归
├── lstm_forecast.py           # LSTM时序预测
├── reports/                   # 生成的图片
└── README.md
```

## 技术栈
- Python 3.12
- pandas, numpy, matplotlib
- statsmodels (ADF, ACF/PACF, ARIMA, VAR, 协整检验)
- arch (GARCH)
- tensorflow (LSTM)

## 作者
qingyangfeng666
