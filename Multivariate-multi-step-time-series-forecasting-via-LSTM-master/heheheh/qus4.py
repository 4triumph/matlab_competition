import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def read_data(file_path):
    df = pd.read_excel(file_path, header=None)
    return df

def create_new_data_frame(df):
    depth_data = df.iloc[1:, 1:].values.astype(float)
    x_data = df.iloc[0, 1:].values * 1852
    y_data = df.iloc[1:, 0].values * 1852

    new_data = []
    for i, y in enumerate(y_data):
        for j, x in enumerate(x_data):
            new_data.append([x, y, depth_data[i, j]])

    new_df = pd.DataFrame(new_data, columns=['x', 'y', 'depth'])
    return new_df

def calculate_d_min_max(depth_data, theta):
    d_min = 2 * np.min(depth_data) * np.tan(theta / 2) * 0.8
    d_max = 2 * np.max(depth_data) * np.tan(theta / 2) * 1.1
    return d_min, d_max

def get_depth_at_point(x_point, y_point, x_data, y_data, depth_data, depth_cache):
    if (x_point, y_point) in depth_cache:
        return depth_cache[(x_point, y_point)]

    points = np.column_stack((x_data.ravel(), y_data.ravel()))
    values = depth_data.ravel()
    depth = griddata(points, values, (x_point, y_point), method='linear')
    return depth

def fitness(chromosome, x_data, y_data, length_m, theta, d_min, d_max, depth_data, depth_cache):
    total_length = 0
    last_line = 0
    for line in chromosome:
        depth = get_depth_at_point(line, length_m / 2, x_data, y_data, depth_data, depth_cache)
        coverage_width = 2 * depth * np.tan(theta / 2)
        overlap = coverage_width - (line - last_line)
        if overlap < d_min or overlap > d_max:
            return 0
        total_length += line - last_line
        last_line = line

    if last_line + coverage_width < width_m:
        return 0

    return 1 / total_length

def select_parents(population, fitness_values):
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

def mutate(chromosome, mutation_rate, d_min, d_max):
    """使用均匀突变"""
    idx = np.random.randint(0, len(chromosome))
    mutation_value = np.random.uniform(-mutation_rate, mutation_rate) * (d_max - d_min)
    chromosome[idx] += mutation_value

def genetic_algorithm_final(population_size, num_generations, mutation_rate, width_m, length_m, theta)):
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

# 主程序
file_path = r'C:\Users\Administrator\Documents\WPSDrive\656140418\WPS云盘\数学建模\附件数据.xlsx'
data_frame = read_data(file_path)
new_df = create_new_data_frame(data_frame)

theta = np.radians(120)
d_min, d_max = calculate_d_min_max(new_df['depth'].values, theta)

depth_cache = {}

width_m = 4 * 1852
length_m = 5 * 1852

width_m1 = 5 * 1852
length_m1 = 54 * 1852

population_size = 100
num_generations = 500
mutation_rate = 0.1

best_solution_final = genetic_algorithm_final(population_size, num_generations, mutation_rate, width_m, length_m, theta)
print(best_solution_final)
best_solution_vertical = genetic_algorithm_final(population_size, num_generations, mutation_rate, width_m1, length_m1, theta)
print(best_solution_vertical)

plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
plt.plot(best_solution_final, best_solution_vertical[:len(best_solution_final)])
plt.xlabel('水平解')
plt.ylabel('垂直解')
plt.title('水平解和垂直解之间的关系')
plt.grid(True)

plt.legend()
plt.show()
