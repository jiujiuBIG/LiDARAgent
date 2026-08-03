import top_rugosity
import numpy as np

# 生成模拟数据
np.random.seed(42)
n = 10000
x = np.random.uniform(0, 50, n)
y = np.random.uniform(0, 50, n)
z = np.random.uniform(0, 20, n) + 5
treeid = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

# 调用函数
result = top_rugosity.compute_top_rugosity(x, y, z, treeid)
print(f"Top Rugosity = {result[0]:.4f} m")