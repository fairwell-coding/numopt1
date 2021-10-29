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
from scipy.optimize import approx_fprime
from typing import Callable

# Modify the following global variables to be used in your functions
""" Start of your code
"""

alpha = 1
beta = 5

d = None
b = None
D = None
A = None

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
    ax[0, 0].annotate("x2 = x1/3 + 5/6", (0, 5/6 + delta), color='red')


# Modify the function bodies below to be used for function value and gradient computation
def approx_grad_task1(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """ Numerical Gradient Computation
        @param x Vector of size (2,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using the given central differences formulation for 2D functions. (Task1 functions)
        @return The gradient approximation
    """
    assert (len(x) == 2)

    epsilon = 10**-4
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
    return x[0]**2 + 5 * x[0] - 6 * x[0] * x[1] + 9 * x[1]**2 - 15 * x[1] + 25/4


def grad_1a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    return [2 * x[0] - 6 * x[1] + 5, - 6 * x[0] + 18 * x[1] - 15]


def func_1b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    return (x[0] - 2)**2 + (x[0]*x[1]**2) - 2


def grad_1b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    return [2*x[0] + x[1]**2 - 4, 2*x[0] * x[1]]


def func_1c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    return x[0]**2 + x[0] * (x[0]**2 + x[1]**2) + (x[0]**2 + x[1]**2)


def grad_1c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    return [3 * x[0]**2 + 4 * x[0] + x[1]**2, 2 * x[0] * x[1] + 2 * x[1]]


def func_1d(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    return alpha * x[0]**2 - 2 * x[0] + beta * x[1]**2


def grad_1d(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    return [2 * alpha * x[0] - 2, 2 * beta * x[1]]


def func_2a(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    pass


def grad_2a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    pass


def func_2b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    pass


def grad_2b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    pass


def func_2c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    pass


def grad_2c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    pass


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

    """ End of your code
    """
    return fig


def task4():
    """ Scheduling Optimization Problem
        @return The scheduling plan M
    """

    """ Start of your code
    """
    M = None
    """ End of your code
    """
    return M


if __name__ == '__main__':
    tasks = [task1, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        pdf.savefig(retval)
    pdf.close()

    task4()
