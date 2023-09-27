from program.Function import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

pd.set_option('expand_frame_repr', False)

# call = {'gamma': [0, 2, 4, 6, 8, 10],
#         'K/S=0.94': [0.4303, 0.2919, 0.05, -0.1606, -0.3540, -0.5285],
#         'K/S=1': [0.6566, 0.3768, -0.0004, -0.2733, -0.4833, -0.6446],
#         'K/S=1.06': [1.1347, 0.4936, -0.1093, -0.4423, -0.6422, -0.7699]}
#
# put = {'gamma': [0, 2, 4, 6, 8, 10],
#        'K/S=0.94': [-0.1698, -0.3920, -0.5725, -0.7034, -0.8065, -0.8727],
#        'K/S=1': [-0.1765, -0.3656, -0.5295, -0.6590, -0.7703, -0.8447],
#        'K/S=1.06': [-0.2177, -0.3459, -0.4797, -0.6036, -0.7231, -0.8067]}
# # 提取数据
# x = put['gamma']
# y1 = put['K/S=0.94']
# y2 = put['K/S=1']
# y3 = put['K/S=1.06']
#
# # 创建画布和坐标轴对象
# fig, ax = plt.subplots()
#
# # 绘制线条和标记
# ax.plot(x, y1, label='K/S=0.94', color='blue', linewidth=1, marker='*')
# ax.plot(x, y2, label='K/S=1', color='red', linewidth=1, marker='^')
# ax.plot(x, y3, label='K/S=1.06', color='green', linewidth=1, marker='o')
#
# # 设置坐标轴标签和标题
# ax.set_xlabel('γ')
# ax.set_ylabel('Expected Option Return')
# ax.set_title('Expected Put Option Returns')
#
# # 添加图例
# ax.legend()
#
# # 显示图形
# plt.show()

stock = pd.read_csv('D:/data/数据/stock-trading-data-pro/sh603663.csv', encoding='gbk', skiprows=1)
stock['交易日期'] = pd.to_datetime(stock['交易日期'])
stock['year'] = stock['交易日期'].dt.year
stock = stock[stock['year'] == 2022]
stock['ret'] = stock['收盘价'] / stock['前收盘价'] - 1
stock['lnret'] = np.log(stock['收盘价'] / stock['前收盘价'])

# 画出收益率走势图
plt.figure(figsize=(14, 7))
plt.plot(stock['交易日期'], stock['ret'], label='Returns')
plt.plot(stock['交易日期'], stock['lnret'], label='Log Returns')
plt.legend()
plt.show()

# 画出收益率直方图
plt.figure(figsize=(14, 7))
plt.hist(stock['ret'].dropna(), bins=50, alpha=0.5, label='Returns')
plt.hist(stock['lnret'].dropna(), bins=50, alpha=0.5, label='Log Returns')
plt.legend()
plt.show()

# 计算20天和60天的波动率
stock['20D Volatility'] = stock['lnret'].rolling(window=20).std() * np.sqrt(252)
stock['60D Volatility'] = stock['lnret'].rolling(window=60).std() * np.sqrt(252)

# 画出20天和60天波动率走势图
plt.figure(figsize=(14, 7))
plt.plot(stock.index, stock['20D Volatility'], label='20D Volatility')
plt.plot(stock.index, stock['60D Volatility'], label='60D Volatility')
plt.legend()
plt.show()