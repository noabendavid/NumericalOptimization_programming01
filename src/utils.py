import matplotlib.pyplot as plt
import numpy as np

def plot_contour_lines(f, x_limits, y_limits, paths={}, levels=100, fig=None,ax=None):
    x = np.linspace(*x_limits, 200)
    y = np.linspace(*y_limits, 200)

    xx, yy = np.meshgrid(x, y)
    func_value = np.vectorize(lambda x1, x2: f(np.array([x1, x2]), False)[0])(xx, yy)

    contour = ax.contour(x, y, func_value, levels, cmap='RdBu_r')
    fig.colorbar(contour, ax=ax)

    ax.set_title('Contour of Objective Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if len(paths) > 0:
        for key, value in paths.items():
            xk = [val[0] for val in value]
            yk = [val[1] for val in value]
            colors = 'darkviolet' if key == 'Newton' else 'deeppink'
            ax.plot(xk, yk, label=key, color=colors)
        ax.legend()

def plot_function_values(func_values, ax=None):
    for key, value in func_values.items():
        x = np.linspace(0, len(value)-1, len(value))
        colors = 'darkviolet' if key == 'Newton' else 'deeppink'
        ax.plot(x, value, label=key, color=colors)
    ax.set_title('Function values vs Iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x)')
    ax.legend()