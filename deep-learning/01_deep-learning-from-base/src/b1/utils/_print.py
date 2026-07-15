from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


def print_truth_table(gate: Callable[[int, int], int]):
    print(f"{gate.__name__} Truth Table")
    print("x1 x2 y")
    for x1 in range(2):
        for x2 in range(2):
            print(f"{x1}  {x2}  {gate(x1, x2)}")


def print_function_graph(
    func: Callable[[float], float], x_range: tuple[float, float], num_points: int = 100
):

    x = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / num_points)
    y = func(x)

    plt.plot(x, y)
    plt.title(f"{func.__name__} Function Graph")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


def print_function_graph_3d(
    func: Callable[[np.ndarray], np.ndarray],
    x1_range: tuple[float, float],
    x2_range: tuple[float, float],
    num_points: int = 100,
    output_index: int | None = 0,
):
    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    x2 = np.linspace(x2_range[0], x2_range[1], num_points)
    X1, X2 = np.meshgrid(x1, x2)

    grid = np.stack([X1, X2], axis=-1)

    points = grid.reshape(-1, 2)

    raw_outputs = [func(point) for point in points]
    outputs = np.asarray(raw_outputs)

    if outputs.ndim == 1:
        Z = outputs.reshape(num_points, num_points)

        ax = plt.axes(projection="3d")
        ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none", alpha=0.7)

    elif outputs.ndim == 2:
        output_dim = outputs.shape[1]
        outputs = outputs.reshape(num_points, num_points, output_dim)

        ax = plt.axes(projection="3d")

        if output_index is None:
            for k in range(output_dim):
                Z = outputs[..., k]
                ax.plot_surface(X1, X2, Z, edgecolor="none", alpha=0.5)
        else:
            Z = outputs[..., output_index]
            ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none", alpha=0.7)

    else:
        raise ValueError(
            f"func output must be scalar or 1D vector per point, got shape {outputs.shape}"
        )
    ax.set_title(f"Pointwise Surface: {func.__name__}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Output Value")

    ax.invert_yaxis()

    plt.show()
