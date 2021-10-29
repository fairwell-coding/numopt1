from main import *

if __name__ == '__main__':
    x = np.random.rand(2)
    print(grad_1a(x))
    print(approx_grad_task1(func_1a, x))
