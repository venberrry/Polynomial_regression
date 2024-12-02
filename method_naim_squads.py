import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def sample():
    x = np.linspace(0, 2 * np.pi, 1000)  # Массив из 1000 точек от 0 до 2π
    y = 100 * np.sin(x) + 0.5 * np.exp(x) + 300 + 10 * np.random.randn(1000)
    return x, y


# Реализация отображения моделей с разной степенью полинома
def polynomial_fit_and_plot(x, y, max_degree=20):
    """
    Обучение полиномиальной регрессии для степеней от 1 до max_degree и визуализация результата.
    """
    fig, axes = plt.subplots(4, 5, figsize=(18, 12))  # 4 строки, 5 столбцов для графиков
    axes = axes.flatten()

    sse_values = []  # для сохранения ошибок

    for degree in range(1, max_degree + 1):
        # построение матрицы признаков (дизайн-матрица)
        X = np.vander(x, degree + 1, increasing=True)

        # наименьшие квадраты
        weights = np.linalg.pinv(X) @ y

        # предикшн
        y_pred = X @ weights

        # Вычисление ошибки SSE
        sse = np.sum((y - y_pred) ** 2)
        sse_values.append(sse)

        # Визуализация результата
        ax = axes[degree - 1]
        ax.scatter(x, y, color='red', s=1, label='Data')  # Данные
        ax.plot(x, y_pred, color='blue', label='Fit')  # Полиномиальная модель
        ax.set_title(f"deg={degree}")
        ax.set_ylim([y.min() - 50, y.max() + 50])

    # Финальная визуализация ошибок SSE
    fig.tight_layout()
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_degree + 1), sse_values, color='red', marker='o', label='SSE')
    plt.title("Error SSE calculation")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("SSE")
    plt.grid()
    plt.legend()
    fig.savefig("img.png")
    plt.savefig("sse_plot.png")


# Генерация данных
x, y = sample()

# Вызов функции для отображения моделей
polynomial_fit_and_plot(x, y)
