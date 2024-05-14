import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r'C:\Users\Administrator\Documents\WPSDrive\656140418\WPS云盘\数学建模\副本output_data(1).xlsx'
df = pd.read_excel(file_path)

# 提取数据
difference_list = df.iloc[1:, 0]
# 打印数据
# for i in range(len(difference_list)):
#     print(difference_list.iloc[i])

print(difference_list)
# 设置每个条形的宽度
width = 0.3
height_list = [1852] * len(difference_list)
xcoor_position = difference_list.tolist()
# 绘制条形图，并为每个条形图指定标签
plt.figure(figsize=(12, 7))

plt.rcParams['font.family'] = 'Courier New'

plt.bar(xcoor_position, height_list)
# plt.bar(x_coordinates, positive_heights, width=width, color="lightblue", label="Positive")
# plt.bar(x_coordinates, negative_heights, width=width, color="lightblue", label="Negative", bottom=positive_heights)
# plt.xticks(rotation=25)  # 设置 x 轴刻度标签为 difference_list 中的值，不再需要 x_coordinates


plt.show()
