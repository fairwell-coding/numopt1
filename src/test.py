from main import func_1b, grad_1b, approx_grad_task1, func_1c, grad_1c

if __name__ == '__main__':
    x = [13, 22]
    print(grad_1b(x))
    print(approx_grad_task1(func_1b, x))
