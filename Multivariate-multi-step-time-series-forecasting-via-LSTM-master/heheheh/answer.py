import griddata as griddata
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# 导入数据
file_path = r'C:\Users\Administrator\Documents\WPSDrive\656140418\WPS云盘\数学建模\附件数据.xlsx'
df = pd.read_excel(file_path, header=None)

# 获取深度坐标
# depth_data = df.values[2:, 2:].astype(float)
depth_data = df.iloc[1:, 1:].values.astype(float)
# 横纵坐标数据的提取并换算
# x_data = df.iloc[1, 2:].values.astype(float)
# y_data = df.iloc[:, 1].values.astype(float)
x_data = df.iloc[0, 1:].values * 1852
y_data = df.iloc[1:, 0].values * 1852

# 创建新的数据集，包含横、纵坐标，深度
new_data = []

for i, y in enumerate(y_data):
    for j, x in enumerate(x_data):
        new_data.append([x, y, depth_data[i, j]])

# 创建新的DataFrame
new_df = pd.DataFrame(new_data, columns=['x', 'y', 'depth'])

# 参数初始化
width_m = 4 * 1852
length_m = 5 * 1852
theta = np.radians(120)

#
# # 最小和最大的测线间距，根据平均海深计算
# avg_depth = np.mean(new_df['depth'].values)
# d_min = 2 * avg_depth * np.tan(theta / 2) * (1 - 0.2)
# d_max = 2 * avg_depth * np.tan(theta / 2) * (1 + 0.2)

# print(d_min, d_max)
# (173.31228684343077 259.96843026514614)

# 确定测线的间距范围。d_min 表示最小的测线间距，而 d_max 表示最大的测线间距，确保测线布局满足指定的条件
d_min, d_max = 2 * np.min(depth_data) * np.tan(theta / 2) * 0.8, 2 * np.max(depth_data) * np.tan(theta / 2) * 1.1
# 2 * np.min(depth_data)：这部分计算了深度数据中的最小值的两倍。np.min(depth_data) 返回深度数据中的最小值。
# np.tan(theta / 2)：这部分计算了扫描角度的一半的正切值。theta 是扫描角度，通过 np.radians(120) 转换为弧度表示，然后除以2来得到一半的角度。
# 0.8：这是一个乘法因子，表示最小间距为深度的80％。
# 2 * np.max(depth_data)：这部分计算了深度数据中的最大值的两倍。np.max(depth_data) 返回深度数据中的最大值。
# np.tan(theta / 2)：这部分计算了扫描角度的一半的正切值。
# 1.1：这是一个乘法因子，表示最大间距为深度的110％。

# 遗传算法参数
population_size = 100
num_generations = 500
# elite_size = 10
mutation_rate = 0.1
# crossover_rate = 0.8

# 创建深度缓存
depth_cache = {}


def get_depth_at_point(x_point, y_point):
    """ 使用线性插值从给定数据点获取深度值"""
    global depth_cache
    if (x_point, y_point) in depth_cache:
        return depth_cache[(x_point, y_point)]

    X, Y = np.meshgrid(x_data, y_data)
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = depth_data.ravel()
    depth = griddata(points, values, (x_point, y_point), method='linear')
    return depth
# def get_depth_at_point(x_point, y_point):
#     """ 使用线性插值从给定数据点获取深度值"""
#     global depth_cache
#     if (x_point, y_point) in depth_cache:
#         return depth_cache[(x_point, y_point)]
#
#     points = np.column_stack((x_data.ravel(), y_data.ravel()))
#     values = depth_data.ravel()
#     depth = griddata(points, values, (x_point, y_point), method='linear')
#     return depth


def get_dmin_dmax(depth):
    """根据深度计算最小和最大的侧线间距"""
    d_min = 2 * depth * np.tan(theta / 2) * (1 - 0.2)
    d_max = 2 * depth * np.tan(theta / 2) * (1 + 0.2)
    return d_min, d_max


def fitness(chromosome):
    """评估染色体的适应度"""
    total_length = 0
    last_line = 0
    for line in chromosome:
        depth = get_depth_at_point(line, length_m / 2)
        coverage_width = 2 * depth * np.tan(theta / 2)
        overlap = coverage_width - (line - last_line)
        d_min, d_max = get_dmin_dmax(depth)
        if overlap < d_min or overlap > d_max:
            return 0
        total_length += line - last_line
        last_line = line

    if last_line + coverage_width < width_m:
        return 0

    return 1 / total_length


def select_parents(population, fitness):
    """使用轮盘赌选择法选择两个父代"""
    idx = np.argsort(fitness)
    sorted_population = population[idx]
    sorted_fitness = np.array(fitness)[idx]
    total_fit = np.sum(sorted_fitness)
    r1 = np.random.rand() * total_fit
    r2 = np.random.rand() * total_fit
    idx1, idx2 = -1, -1

    for i, f, in enumerate(sorted_fitness):
        r1 -= f
        if r1 <= 0:
            idx1 = i
            break

    for i, f, in enumerate(sorted_fitness):
        r2 -= f
        if r2 <= 0:
            idx2 = i
            break
    # 符号可能是.
    return sorted_population[idx1], sorted_population[idx2]


def crossover(parent1, parent2):
    """使用单点交叉"""
    idx = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:idx], parent2[idx:]))
    child2 = np.concatenate((parent2[:idx], parent1[idx:]))
    return child1, child2


def mutate(chromosome):
    """使用均匀突变"""
    idx = np.random.randint(0, len(chromosome))
    mutation_value = np.random.uniform(-mutation_rate, mutation_rate) * (d_max - d_min)
    chromosome[idx] += mutation_value


def genetic_algorithm_final():
    d_avg = (d_min + d_max) / 2
    num_lines = int(width_m / d_avg)
    population = []
    for _ in range(population_size):
        start = np.random.uniform(0, d_avg)
        chromosome = [start + i * d_avg for i in range(num_lines)]
        population.append(chromosome)
    population = np.array(population)

    for generation in range(num_generations):
        fitness_values = [fitness(chromo) for chromo in population]

        new_population = []
        for i in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            if np.random.rand() < mutation_rate:
                mutate(child1)
            if np.random.rand() < mutation_rate:
                mutate(child2)
            new_population.extend([child1, child2])
        population = np.array(new_population)

    best_idx = np.argmax(fitness_values)
    return population[best_idx]


# 开始运行
best_solution_final = genetic_algorithm_final()
print(best_solution_final)

#获取反向思考的解
depth_data1 = df.iloc[1:, 1:].values.astype(float).T
# 重新获取横纵坐标和深度数据，交换x,y的角色
# x_data = df.iloc[1, 2:].values.astype(float)
# y_data = df.iloc[:, 1].values.astype(float)
y_data1 = df.iloc[0, 1:].values * 1852
x_data1 = df.iloc[1:, 0].values * 1852

# 创建新的数据集，包含横、纵坐标，深度
new_data1 = []

for i, y in enumerate(y_data1):
    for j, x in enumerate(x_data1):
        new_data.append([x, y, depth_data1.iloc[i, j]])

# 创建新的DataFrame
new_df1 = pd.DataFrame(new_data1, columns=['x', 'y', 'depth'])
width_m1 = 5 * 1852
length_m1 = 4 * 1852

# 确定测线的间距范围。d_min 表示最小的测线间距，而 d_max 表示最大的测线间距，确保测线布局满足指定的条件
d_min1, d_max1 = 2 * np.min(depth_data1) * np.tan(theta / 2) * 0.8, 2 * np.max(depth_data1) * np.tan(theta / 2) * 1.1
# 2 * np.min(depth_data)：这部分计算了深度数据中的最小值的两倍。np.min(depth_data) 返回深度数据中的最小值。
# np.tan(theta / 2)：这部分计算了扫描角度的一半的正切值。theta 是扫描角度，通过 np.radians(120) 转换为弧度表示，然后除以2来得到一半的角度。
# 0.8：这是一个乘法因子，表示最小间距为深度的80％。
# 2 * np.max(depth_data)：这部分计算了深度数据中的最大值的两倍。np.max(depth_data) 返回深度数据中的最大值。
# np.tan(theta / 2)：这部分计算了扫描角度的一半的正切值。
# 1.1：这是一个乘法因子，表示最大间距为深度的110％。

# 创建深度缓存
depth_cache1 = {}


# def get_depth_at_point(x_point, y_point):
#     """ 使用线性插值从给定数据点获取深度值"""
#     global depth_cache
#     if (x_point, y_point) in depth_cache:
#         return depth_cache[(x_point, y_point)]
#
#     X, Y = np.meshgrid(x_data, y_data)
#     points = np.column_stack((X.ravel(), Y.ravel()))
#     values = depth_data.ravel()
#     depth = griddata(points, values, (x_point, y_point), method='linear')
#     return depth
def get_depth_at_point1(x_point, y_point):
    """ 使用线性插值从给定数据点获取深度值"""
    global depth_cache1
    if (x_point, y_point) in depth_cache1:
        return depth_cache1[(x_point, y_point)]

    points = np.column_stack((x_data1.ravel(), y_data1.ravel()))
    values = depth_data1.ravel()
    depth = griddata(points, values, (x_point, y_point), method='linear')
    return depth


def get_dmin_dmax1(depth):
    """根据深度计算最小和最大的侧线间距"""
    d_min1 = 2 * depth * np.tan(theta / 2) * (1 - 0.2)
    d_max1 = 2 * depth * np.tan(theta / 2) * (1 + 0.2)
    return d_min1, d_max1


def fitness1(chromosome):
    """评估染色体的适应度"""
    total_length = 0
    last_line = 0
    for line in chromosome:
        depth = get_depth_at_point1(line, length_m / 2)
        coverage_width = 2 * depth * np.tan(theta / 2)
        overlap = coverage_width - (line - last_line)
        d_min1, d_max1 = get_dmin_dmax(depth)
        if overlap < d_min or overlap > d_max:
            return 0
        total_length += line - last_line
        last_line = line

    if last_line + coverage_width < width_m:
        return 0

    return 1 / total_length


def select_parents1(population, fitness):
    """使用轮盘赌选择法选择两个父代"""
    idx = np.argsort(fitness)
    sorted_population = population[idx]
    sorted_fitness = np.array(fitness)[idx]
    total_fit = np.sum(sorted_fitness)
    r1 = np.random.rand() * total_fit
    r2 = np.random.rand() * total_fit
    idx1, idx2 = -1, -1

    for i, f, in enumerate(sorted_fitness):
        r1 -= f
        if r1 <= 0:
            idx1 = i
            break

    for i, f, in enumerate(sorted_fitness):
        r2 -= f
        if r2 <= 0:
            idx2 = i
            break
    # 符号可能是.
    return sorted_population[idx1], sorted_population[idx2]


def crossover1(parent1, parent2):
    """使用单点交叉"""
    idx = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:idx], parent2[idx:]))
    child2 = np.concatenate((parent2[:idx], parent1[idx:]))
    return child1, child2


def mutate1(chromosome):
    """使用均匀突变"""
    idx = np.random.randint(0, len(chromosome))
    mutation_value = np.random.uniform(-mutation_rate, mutation_rate) * (d_max1 - d_min1)
    chromosome[idx] += mutation_value


def genetic_algorithm_final1():
    d_avg = (d_min1 + d_max1) / 2
    num_lines = int(width_m1 / d_avg)
    population = []
    for _ in range(population_size):
        start = np.random.uniform(0, d_avg)
        chromosome = [start + i * d_avg for i in range(num_lines)]
        population.append(chromosome)
    population = np.array(population)

    for generation in range(num_generations):
        fitness_values = [fitness(chromo) for chromo in population]

        new_population = []
        for i in range(population_size // 2):
            parent1, parent2 = select_parents1(population, fitness_values)
            child1, child2 = crossover1(parent1, parent2)
            if np.random.rand() < mutation_rate:
                mutate1(child1)
            if np.random.rand() < mutation_rate:
                mutate1(child2)
            new_population.extend([child1, child2])
        population = np.array(new_population)

    best_idx = np.argmax(fitness_values)
    return population[best_idx]


# 开始运行
best_solution_vertical = genetic_algorithm_final()
print(best_solution_vertical)


plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
plt.plot(best_solution_final, best_solution_vertical[:len(best_solution_final)])
plt.xlabel('水平解')
plt.ylabel('垂直解')
plt.title('水平解和垂直解之间的关系')
plt.grid(True)
plt.show()