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

alpha = None
beta = None

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

    fig, ax = plt.subplots(2, 2, figsize=(12,12))
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
    ax[0, 0].contour(x1, x2, x1**2 + 0.5*x2**2, 50)

    """ End of your code
    """
    return fig

# Modify the function bodies below to be used for function value and gradient computation
def approx_grad_task1(func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    
    """ Numerical Gradient Computation
        @param x Vector of size (2,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using the given central differences formulation for 2D functions. (Task1 functions)
        @return The gradient approximation
    """
    assert(len(x) == 2)
    pass

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
    pass

def grad_1a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    pass

def func_1b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    pass

def grad_1b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    pass

def func_1c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    pass

def grad_1c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    pass

def func_1d(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    pass

def grad_1d(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1d) at a given point x
        @param x Vector of size (2,)
    """
    pass

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
    fig = plt.figure(figsize=(15,10), constrained_layout=True)
    fig.suptitle('Task 3 - Barplots numerical vs analytical', fontsize=16)
    ax = [None, None, None, None, None, None, None]
    keys = ['a)', 'b)', 'c)', 'd)']
    gs = fig.add_gridspec(7, 12)

    n = 2
    for i in range(4):
        ax[i] = fig.add_subplot(gs[1:4, 3*i:(3*i+3)])
        ax[i].set_title('1 '+keys[i])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$'), fontsize=16)

    n = 5
    for k, i in enumerate(range(4, 7)):
        ax[i] = fig.add_subplot(gs[4:, 4*k:(4*k+4)])
        ax[i].set_title('2 '+ keys[k])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$', r'$\frac{\partial}{\partial x_3}$', r'$\frac{\partial}{\partial x_4}$', r'$\frac{\partial}{\partial x_5}$'), fontsize=16)

    """ Start of your code
    """

    # Example for plot usage
    bw = 0.3
    ax[0].bar([0-bw/2,1-bw/2], [1.5, 1.1], bw)
    ax[0].bar([0+bw/2,1+bw/2], [1.5, 1.1], bw)

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