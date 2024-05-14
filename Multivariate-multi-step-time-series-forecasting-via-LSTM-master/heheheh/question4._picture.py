import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 读取海水深度数据
file_path = r'C:\Users\Administrator\Desktop\CUMCM2023Problems-1\B题\附件.xlsx'
df = pd.read_excel(file_path, header=None)
depth_data = df.values[2:, 2:].astype(float)  # 去掉行和列的标签数据


# 海域尺寸
width = 4  # 东西宽度（海里）
length = 5  # 南北长度（海里）


# 海域网格尺寸
grid_size_x = len(depth_data[0])
grid_size_y = len(depth_data)

# 将每个 z 轴坐标的数值乘以 -1
depth_data *= -1

# 生成横纵坐标数据
x_data = np.linspace(0, width, grid_size_x)
y_data = np.linspace(0, length, grid_size_y)


# 生成 x 和 y 坐标网格
x_mesh, y_mesh = np.meshgrid(x_data, y_data)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 绘制海底水深数据二维图
plt.figure(figsize=(10, 6))
plt.contourf(x_mesh, y_mesh, depth_data, cmap='Blues_r', levels=20)
plt.colorbar(label='海水深度')
plt.xlabel('横向坐标（海里）')
plt.ylabel('纵向坐标（海里）')


# 设置二维图坐标比例
plt.gca().set_aspect('equal', adjustable='box')

plt.show()


# 绘制海水深度数据三维图


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_mesh, y_mesh, depth_data, cmap='Blues_r')

ax.set_xlabel('横向坐标（海里）')
ax.set_ylabel('纵向坐标（海里）')
ax.set_zlabel('海水深度（米）')

plt.show()
