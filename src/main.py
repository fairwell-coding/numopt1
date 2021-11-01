#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import approx_fprime, linprog
from typing import Callable

# Modify the following global variables to be used in your functions
""" Start of your code
"""

alpha = 1
beta = 5
epsilon = 10 ** -4  # used for tasks 3.1, 3.2 and 4

d = 2.5
n_dim = 5
b = np.random.rand(n_dim)
D = np.matrix(np.random.rand(n_dim, n_dim))
A = np.matrix(np.random.rand(n_dim, n_dim))

""" End of your code
"""


def task1():
    """ Characterization of Functions

        Requirements for the plots:
            - ax[0, 0] Contour plot for a)
            - ax[0, 1] Contour plot for b)
            - ax[1, 0] Contour plot for c)
            - ax[1, 1] Contour plot for d)
    """

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Task 1 - Contour plots of functions', fontsize=16)

    ax[0, 0].set_title('a)')
    ax[0, 0].set_xlabel('$x_1$')
    ax[0, 0].set_ylabel('$x_2$')

    ax[0, 1].set_title('b)')
    ax[0, 1].set_xlabel('$x_1$')
    ax[0, 1].set_ylabel('$x_2$')

    ax[1, 0].set_title('c)')
    ax[1, 0].set_xlabel('$x_1$')
    ax[1, 0].set_ylabel('$x_2$')

    ax[1, 1].set_title('d)')
    ax[1, 1].set_xlabel('$x_1$')
    ax[1, 1].set_ylabel('$x_2$')

    """ Start of your code
    """

    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    __plot_1a(ax, x1, x2)
    __plot_1b(ax, x1, x2)
    __plot_1c(ax, x1, x2)
    __plot_1d(ax, x1, x2)

    """ End of your code
    """
    return fig


def __plot_1d(ax, x1, x2):
    ax[1, 1].contour(x1, x2, alpha * x1 ** 2 - 2 * x1 + beta * x2 ** 2, 50)
    ax[1, 1].plot(1 / alpha, 0, color='red', marker="x", markersize=8)
    delta = 0.2
    ax[1, 1].annotate("P(1/alpha| 0)", (0 + delta, 0 + delta), color='red')


def __plot_1c(ax, x1, x2):
    ax[1, 0].contour(x1, x2, x1 ** 2 + x1 * (x1 ** 2 + x2 ** 2) ** 0.5 + (x1 ** 2 + x2 ** 2) ** 0.5, 50)
    delta = 0.2

    ax[1, 0].plot(-1, 1, color='red', marker="x", markersize=8)
    ax[1, 0].annotate("P1(-1|1) = saddle point", (-1 + delta, 1 + delta), color='red')

    ax[1, 0].plot(-1, -1, color='red', marker="x", markersize=8)
    ax[1, 0].annotate("P2(-1|-1) = saddle point", (-1 + delta, -1 + delta), color='red')

    ax[1, 0].plot(0, 0, color='red', marker="x", markersize=8)
    ax[1, 0].annotate("P3(0|0) = local minima", (0 + delta, 0 + delta), color='red')


def __plot_1b(ax, x1, x2):
    ax[0, 1].contour(x1, x2, (x1 - 2) ** 2 + x1 * x2 ** 2 - 2, 50)

    delta = 0.2

    ax[0, 1].plot(2, 0, color='red', marker="x", markersize=8)
    ax[0, 1].annotate("P1(2|0)", (2 + delta, 0 + delta), color='red')

    ax[0, 1].plot(0, 2, color='red', marker="x", markersize=8)
    ax[0, 1].annotate("P2(0|2)", (0 + delta, 2 + delta), color='red')

    ax[0, 1].plot(0, -2, color='red', marker="x", markersize=8)
    ax[0, 1].annotate("P3(0|-2)", (0 + delta, -2 + delta), color='red')


def __plot_1a(ax, x1, x2):
    ax[0, 0].contour(x1, x2, (-x1 + 3 * x2 - 2.5) ** 2, 50)
    # The critical point is undefined (det(Hessian) = 0). The solution of the respective linear system has one degree of freedom, hence for the critical point we get the line: x2 = x1/3 + 5/6
    x = np.linspace(-5, 5)
    y = x / 3 + 5 / 6
    ax[0, 0].plot(x, y, color='red')
    delta = 1
    ax[0, 0].annotate("x2 = x1/3 + 5/6", (0, 5 / 6 + delta), color='red')


# Modify the function bodies below to be used for function value and gradient computation
def approx_grad_task1(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """ Numerical Gradient Computation
        @param x Vector of size (2,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using the given central differences formulation for 2D functions. (Task1 functions)
        @return The gradient approximation
    """
    assert (len(x) == 2)

    grad_1 = func([x[0] + epsilon, x[1]]) - func([x[0] - epsilon, x[1]])
    grad_2 = func([x[0], x[1] + epsilon]) - func([x[0], x[1] - epsilon])

    return np.multiply([grad_1, grad_2], 1 / (2 * epsilon))


def approx_grad_task2(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """ Numerical Gradient Computation
        @param x Vector of size (n,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using scipy.optimize.approx_fprime(). (Task2 functions)
        @return The gradient approximation
    """
    pass


def func_1a(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    return x[0] ** 2 + 5 * x[0] - 6 * x[0] * x[1] + 9 * x[1] ** 2 - 15 * x[1] + 25 / 4


def grad_1a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    return [2 * x[0] - 6 * x[1] + 5, - 6 * x[0] + 18 * x[1] - 15]


def func_1b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    return (x[0] - 2) ** 2 + (x[0] * x[1] ** 2) - 2


def grad_1b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    return [2 * x[0] + x[1] ** 2 - 4, 2 * x[0] * x[1]]


def func_1c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    return x[0] ** 2 + x[0] * (x[0] ** 2 + x[1] ** 2) + (x[0] ** 2 + x[1] ** 2)


def grad_1c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    return [3 * x[0] ** 2 + 4 * x[0] + x[1] ** 2, 2 * x[0] * x[1] + 2 * x[1]]


def func_1d(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    return alpha * x[0] ** 2 - 2 * x[0] + beta * x[1] ** 2


def grad_1d(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    return [2 * alpha * x[0] - 2, 2 * beta * x[1]]


def func_2a(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    return 1 / 4 * sum((x - b) ** 2) ** 2


def grad_2a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    return (x - b) * sum((x - b) ** 2)


def func_2b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    return sum(1 / 2 * np.multiply(A.dot(np.asmatrix(x).T), A.dot(np.asmatrix(x).T)) + A.dot(np.asmatrix(x).T))[0, 0]


def grad_2b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    return np.asarray((A.T * (A * np.matrix(x).T + np.matrix([1, 1, 1, 1, 1]).T))).squeeze()


def func_2c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    return (np.asmatrix(np.divide(x, b)) * D * np.asmatrix(np.divide(x, b)).T)[0, 0]  # (x hadmard b)^T * D * (x hadmard b)


def grad_2c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    return np.asarray(np.divide((D.T + D) * np.asmatrix(np.divide(x, b)).T, np.asmatrix(b).T)).squeeze()


def task3():
    """ Numerical Gradient Verification
        ax[0] to ax[3] Bar plot comparison, analytical vs numerical gradient for Task 1
        ax[4] to ax[6] Bar plot comparison, analytical vs numerical gradient for Task 2

    """
    n = 5
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    fig.suptitle('Task 3 - Barplots numerical vs analytical', fontsize=16)
    ax = [None, None, None, None, None, None, None]
    keys = ['a)', 'b)', 'c)', 'd)']
    gs = fig.add_gridspec(7, 12)

    n = 2
    for i in range(4):
        ax[i] = fig.add_subplot(gs[1:4, 3 * i:(3 * i + 3)])
        ax[i].set_title('1 ' + keys[i])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$'), fontsize=16)

    n = 5
    for k, i in enumerate(range(4, 7)):
        ax[i] = fig.add_subplot(gs[4:, 4 * k:(4 * k + 4)])
        ax[i].set_title('2 ' + keys[k])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels(
            (r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$', r'$\frac{\partial}{\partial x_3}$', r'$\frac{\partial}{\partial x_4}$', r'$\frac{\partial}{\partial x_5}$'),
            fontsize=16)

    """ Start of your code
    """

    __plot_solutions_task1(ax)
    __plot_solutions_task2(ax)

    """ End of your code
    """
    return fig


def __plot_solutions_task2(ax):
    x = np.random.randn(5)

    width = 0.3
    del_x1_analytical = 0 - width / 2
    del_x2_analytical = 1 - width / 2
    del_x3_analytical = 2 - width / 2
    del_x4_analytical = 3 - width / 2
    del_x5_analytical = 4 - width / 2

    del_x1_numerical = 0 + width / 2
    del_x2_numerical = 1 + width / 2
    del_x3_numerical = 2 + width / 2
    del_x4_numerical = 3 + width / 2
    del_x5_numerical = 4 + width / 2

    analytical_bars = [del_x1_analytical, del_x2_analytical, del_x3_analytical, del_x4_analytical, del_x5_analytical]
    numerical_bars = [del_x1_numerical, del_x2_numerical, del_x3_numerical, del_x4_numerical, del_x5_numerical]

    ax[4].bar(analytical_bars, grad_2a(x), width)
    ax[4].bar(numerical_bars, approx_fprime(x, func_2a, epsilon), width)

    ax[5].bar(analytical_bars, grad_2b(x), width)
    ax[5].bar(numerical_bars, approx_fprime(x, func_2b, epsilon), width)

    ax[6].bar(analytical_bars, grad_2c(x), width)
    ax[6].bar(numerical_bars, approx_fprime(x, func_2c, epsilon), width)


def __plot_solutions_task1(ax):
    x = np.random.randn(2)

    width = 0.3
    del_x1_analytical = 0 - width / 2
    del_x2_analytical = 1 - width / 2
    del_x1_numerical = 0 + width / 2
    del_x2_numerical = 1 + width / 2
    analytical_bars = [del_x1_analytical, del_x2_analytical]
    numerical_bars = [del_x1_numerical, del_x2_numerical]

    ax[0].bar(analytical_bars, [grad_1a(x)[0], grad_1a(x)[1]], width)
    ax[0].bar(numerical_bars, [approx_grad_task1(func_1a, x)[0], approx_grad_task1(func_1a, x)[1]], width)

    ax[1].bar(analytical_bars, [grad_1b(x)[0], grad_1b(x)[1]], width)
    ax[1].bar(numerical_bars, [approx_grad_task1(func_1b, x)[0], approx_grad_task1(func_1b, x)[1]], width)

    ax[2].bar(analytical_bars, [grad_1c(x)[0], grad_1c(x)[1]], width)
    ax[2].bar(numerical_bars, [approx_grad_task1(func_1c, x)[0], approx_grad_task1(func_1c, x)[1]], width)

    ax[3].bar(analytical_bars, [grad_1d(x)[0], grad_1d(x)[1]], width)
    ax[3].bar(numerical_bars, [approx_grad_task1(func_1d, x)[0], approx_grad_task1(func_1d, x)[1]], width)


def task4():
    """ Scheduling Optimization Problem
        @return The scheduling plan M
    """

    """ Start of your code
    """

    cpu = np.asarray([0.11, 0.13, 0.09, 0.12, 0.15, 0.14, 0.11, 0.12])
    gpu = np.asarray([0.10, 0.13, 0.08, 0.13, 0.14, 0.14, 0.09, 0.13])
    energy = np.concatenate([cpu, gpu])

    A_ub = np.asarray([[0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x3 <= 0
                       [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x4 <= 0
                       [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x5 <= 0
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x6 <= 0
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # -x7 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # -y0 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # -y1 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # -y2 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # -y3 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],  # -y4 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # -y5 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # -y6 <= 0
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],  # -y7 <= 0
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x0 <= -480
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x1 <= -600
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x2 <= -560
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # x0 + .. + x7 <= 4500
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # y0 + .. + y7 <= 4500
                       ])

    b_ub = np.asarray([0,  # -x3 <= 0
                       0,  # -x4 <= 0
                       0,  # -x5 <= 0
                       0,  # -x6 <= 0
                       0,  # -x7 <= 0
                       0,  # -y0 <= 0
                       0,  # -y1 <= 0
                       0,  # -y2 <= 0
                       0,  # -y3 <= 0
                       0,  # -y4 <= 0
                       0,  # -y5 <= 0
                       0,  # -y6 <= 0
                       0,  # -y7 <= 0
                       -480,  # -x0 <= -480
                       -600,  # -x1 <= -600
                       -560,  # -x2 <= -560
                       4500,  # x0 + .. + x7 <= 4500
                       4500  # y0 + .. + y7 <= 4500
                       ])

    A_eq = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # x0 + y0 = 1200
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # x1 + y1 = 1500
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # x2 + y2 = 1400
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # x3 + y3 = 400
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # x4 + y4 = 1000
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x5 + y5 = 800
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # x6 + y6 = 760
                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # x7 + y7 = 1300
                       ])

    b_eq = np.asarray([1200,  # x0 + y0 = 1200
                       1500,  # x1 + y1 = 1500
                       1400,  # x2 + y2 = 1400
                       400,  # x3 + y3 = 400
                       1000,  # x4 + y4 = 1000
                       800,  # x5 + y5 = 800
                       760,  # x6 + y6 = 760
                       1300  # x7 + y7 = 1300
                       ])

    optimization_result = linprog(c=energy, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    M = np.asmatrix([optimization_result['x'][:8], optimization_result['x'][8:]]).T

    __verify_optimization_result(M)
    __compute_total_energy_consumption(energy, optimization_result)
    __find_optimal_result_for_task_g_constraints(A_eq, A_ub, b_eq, b_ub, energy)

    """ End of your code
    """
    return M


def __find_optimal_result_for_task_g_constraints(A_eq, A_ub, b_eq, b_ub, energy):
    b_ub[13] = -1200
    b_ub[14] = -1500
    b_ub[15] = -1400
    b_ub[4] = -1300

    task_g_optimization_result = linprog(c=energy, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')

    print('The result is infeasible (result = "{}").'.format(task_g_optimization_result['success']))  # Explanation: Executing processes P0, P1, P2 and P7 entirely on the CPU results in a total
    # number of instructions 1200 + 1500 + 1400 + 1300 = 5400 which is bigger than one of your original constraints, i.e. x0 + ... + x7 <= 4500 instructions. Hence, no optimal solution can be
    # computed without modifying the constraints.


def __compute_total_energy_consumption(energy, optimization_result):
    print('The total energy consumption for the computed scheduling plan is {} uWh.'.format(sum(energy * optimization_result['x'])))


def __verify_optimization_result(M: np.matrix):
    assert M[0, 0] + M[0, 1] == np.round(1200, decimals=5)  # x0 + y0 = 1200
    assert M[1, 0] + M[1, 1] == np.round(1500, decimals=5)  # x1 + y1 = 1500
    assert M[2, 0] + M[2, 1] == np.round(1400, decimals=5)  # x1 + y1 = 1400
    assert M[3, 0] + M[3, 1] == np.round(400, decimals=5)  # x1 + y1 = 400
    assert M[4, 0] + M[4, 1] == np.round(1000, decimals=5)  # x1 + y1 = 1000
    assert M[5, 0] + M[5, 1] == np.round(800, decimals=5)  # x1 + y1 = 800
    assert M[6, 0] + M[6, 1] == np.round(760, decimals=5)  # x1 + y1 = 760
    assert M[7, 0] + M[7, 1] == np.round(1300, decimals=5)  # x1 + y1 = 1300
    assert M[0, 0] >= np.round(480, decimals=5)  # x0 >= 480
    assert M[1, 0] >= np.round(600, decimals=5)  # x1 >= 600
    assert M[2, 0] >= np.round(560, decimals=5)  # x2 >= 560
    assert M[3, 0] >= np.round(0, decimals=5)  # x3 >= 0
    assert M[4, 0] >= np.round(0, decimals=5)  # x4 >= 0
    assert M[5, 0] >= np.round(0, decimals=5)  # x5 >= 0
    assert M[6, 0] >= np.round(0, decimals=5)  # x6 >= 0
    assert M[7, 0] >= np.round(0, decimals=5)  # x7 >= 0
    assert M[0, 1] >= np.round(0, decimals=5)  # y1 >= 0
    assert M[2, 1] >= np.round(0, decimals=5)  # y2 >= 0
    assert M[3, 1] >= np.round(0, decimals=5)  # y3 >= 0
    assert M[4, 1] >= np.round(0, decimals=5)  # y4 >= 0
    assert M[5, 1] >= np.round(0, decimals=5)  # y5 >= 0
    assert M[6, 1] >= np.round(0, decimals=5)  # y6 >= 0
    assert M[7, 1] >= np.round(0, decimals=5)  # y7 >= 0
    assert M[0, 0] + M[1, 0] + M[2, 0] + M[3, 0] + M[4, 0] + M[5, 0] + M[6, 0] + M[7, 0] <= np.round(4500, decimals=5)  # x0 + .. + x7 <= 4500
    assert M[0, 1] + M[1, 1] + M[2, 1] + M[3, 1] + M[4, 1] + M[5, 1] + M[6, 1] + M[7, 1] <= np.round(4500, decimals=5)  # y0 + .. + y7 <= 4500


if __name__ == '__main__':
    tasks = [task1, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        pdf.savefig(retval)
    pdf.close()

    task4()
