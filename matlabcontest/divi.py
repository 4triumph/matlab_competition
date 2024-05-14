# -*- coding: utf-8 -*-
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 读取CSV文件
data = pd.read_csv('附件1-鄱阳湖A水文站逐日水位数据.csv', parse_dates=['时间'], index_col='时间', encoding='gbk')

# 进行时间序列分解
result = sm.tsa.seasonal_decompose(data['数据'], model='additive')

# 设置全局字体
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
title_font = {'fontname': 'SimHei'}

# 绘制原始数据
plt.subplot(4, 1, 1)
plt.plot(data['数据'])
plt.title('原始数据', **title_font)

# 绘制长期趋势
plt.subplot(4, 1, 2)
plt.plot(result.trend)
plt.title('长期趋势', **title_font)

# 绘制季节变动
plt.subplot(4, 1, 3)
plt.plot(result.seasonal)
plt.title('季节变动', **title_font)

# 绘制随机波动
plt.subplot(4, 1, 4)
plt.plot(result.resid)
plt.title('随机波动', **title_font)

plt.tight_layout()
plt.show()