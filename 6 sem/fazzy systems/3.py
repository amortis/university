import numpy as np
import matplotlib.pyplot as plt

# Определим универсум
x = np.linspace(0, 10, 1000)

# Функции принадлежности
def excellent_quality(x):
    return np.where(x >= 7, 1, np.where(x >= 5, (x - 5) / 2, 0))

def high_income_and_prices(x):
    return np.where(x >= 6, 1, np.where(x >= 4, (x - 4) / 2, 0))

def high_profit(x):
    return np.where(x >= 8, 1, np.where(x >= 5, (x - 5) / 3, 0))

def approximately_equal(x, a=5):
    return np.maximum(0, 1 - np.abs(x - a) / 1.5)

def in_interval(x, a=3, b=7):
    return np.where((x >= a) & (x <= b), 1, 0)

def low_quality(x):
    return np.where(x <= 3, 1, np.where(x <= 5, (5 - x) / 2, 0))

def insignificant_value(x):
    return np.where(x <= 2, 1, np.where(x <= 4, (4 - x) / 2, 0))

def significant_value(x):
    return np.where(x >= 8, 1, np.where(x >= 6, (x - 6) / 2, 0))

def average_income_and_prices(x):
    return np.where(x <= 3, 0,
                    np.where(x <= 5, (x - 3) / 2,
                             np.where(x <= 7, (7 - x) / 2, 0)))

# Построим графики
fig, axs = plt.subplots(3, 3, figsize=(18, 12))

functions = [
    (excellent_quality, "a) Отличное качество"),
    (high_income_and_prices, "b) Высокий доход и цены"),
    (high_profit, "c) Высокая норма прибыли"),
    (approximately_equal, "d) Приблизительно равно (a=5)"),
    (in_interval, "e) В интервале [3, 7]"),
    (low_quality, "f) Низкое качество"),
    (insignificant_value, "g) Незначительная величина"),
    (significant_value, "h) Значительная величина"),
    (average_income_and_prices, "i) Средний доход и цены"),
]

for ax, (func, title) in zip(axs.flatten(), functions):
    ax.plot(x, func(x))
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

plt.tight_layout()
plt.show()

