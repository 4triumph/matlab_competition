import math


# 定义位置信息的类
class Position:
    def __init__(self):
        self.x = 0.0
        self.w = 0.0
        self.depth = 0.0
        self.fraction_coverage = 0.0


# 根据题目的得到的条件
alpha = 1.5
theta = 120
PI = 180
D = 70
original_x = D / math.tan(math.radians(alpha))
w1_angle = (PI - theta) / 2 - alpha
w2_angle = (PI - theta) / 2 + alpha
w3_angle = PI - w1_angle - alpha
w4_angle = PI - w2_angle
w5_angle = w2_angle - alpha


# 计算海水深度
def cal_depth(x):
    return ((original_x - x) * D) / original_x

# 计算左侧覆盖宽度
def cal_w1(x):
    return  x * math.sin(math.radians(theta / 2)) / math.sin(math.radians(w1_angle))

# 计算右侧覆盖宽度
def cal_w2(x):
    return x * math.sin(math.radians(theta / 2)) / math.sin(math.radians(w2_angle))


def solve():
    positions = [Position() for _ in range(9)]

    # 初始化测线距中心点处的距离
    for i in range(9):
        positions[i].x = (-4 + i) * 200.0

    # 计算海水深度、覆盖宽度
    for i in range(9):
        if i == 4:
            positions[i].depth = D
            w1 = cal_w1(D)
            w2 = cal_w2(D)
            positions[i].w = w1 + w2
        else:
            positions[i].depth = cal_depth(positions[i].x)
            w1 = cal_w1(positions[i].depth)
            w2 = cal_w2(positions[i].depth)
            positions[i].w = w1 + w2

    # 计算与前一条线的重叠率
    for i in range(1, 9, 1):
        diff1 = (positions[i].x - positions[i - 1].x) * math.sin(math.radians(w5_angle)) / math.sin(
            math.radians(w4_angle))
        diff2 = positions[i].w - diff1
        radio = diff2 / positions[i].w
        positions[i].fraction_coverage = radio * 100

    for i in range(9):
        print(f"测线距中心点处的距离：{positions[i].x}")
        print(f"海水深度：{positions[i].depth}")
        print(f"覆盖宽度：{positions[i].w}")
        if i == 0:
            print("与前一条线的重叠率：————")
        else:
            print(f"与前一条线的重叠率：{positions[i].fraction_coverage}%")


if __name__ == "__main__":
    solve()
