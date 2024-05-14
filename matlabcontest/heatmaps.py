# import csv
# import pandas as pd
# import calplot
# from matplotlib import pyplot as plt
# import numpy as np
#
# data = []
# with open('附件1-鄱阳湖A水文站逐日水位数据.csv', 'r', encoding='gbk') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         data.append(row)
#
# df = pd.DataFrame(data[1:], columns=data[0])
#
# date_column = '时间'
# value_column = '数据'
#
# df[date_column] = pd.to_datetime(df[date_column])
# df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
#
# df.fillna(0, inplace=True)  # 将NaN填充为零
#
# df.set_index(date_column, inplace=True)
#
# calplot.calplot(df[value_column], color='YLGn')  # 指定颜色为绿色
#
# plt.title('时间序列日历热图')
# plt.xlabel('时间')
# plt.ylabel('数据')
#
# plt.show()
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import csv

data = []
with open('附件1-鄱阳湖A水文站逐日水位数据.csv', 'r', encoding='gbk') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

df = pd.DataFrame(data[1:], columns=data[0])

date_column = '时间'
value_column = '数据'

df[date_column] = pd.to_datetime(df[date_column])
df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

df.fillna(0, inplace=True)  # 将NaN填充为零

df.set_index(date_column, inplace=True)

# 设置全局字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei'

# 绘制折线图
plt.plot(df[value_column])
plt.title('时间序列折线图')
plt.xlabel('时间')
plt.ylabel('数据')

plt.show()
