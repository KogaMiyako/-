import numpy as np
import sympy


def test_function():
    # 定义变量
    x1, x2, x3, x4 = sympy.symbols('x1 x2 x3 x4')
    x = [x1, x2, x3, x4]

    # 定义函数
    f1 = x1 + 10 * x2
    f2 = sympy.sqrt(5) * (x3 - x4)
    f3 = (x2 - 2 * x3) ** 2
    f4 = sympy.sqrt(10) * (x1 - x4) ** 2
    f = f1 ** 2 + f2 ** 2 + f3 ** 2 + f4 ** 2

    # 求解梯度
    grad_f = [sympy.lambdify((x1, x2, x3, x4), sympy.diff(f, xi), 'numpy') for xi in x]

    # 求解海森矩阵
    hessian_f = np.zeros((4, 4), dtype=object)
    for i in range(4):
        for j in range(4):
            # hessian_f[i][j] = sympy.diff(sympy.diff(f, x[i]), x[j])
            hessian_f[i][j] = sympy.lambdify((x1, x2, x3, x4), sympy.diff(sympy.diff(f, x[i]), x[j]), 'numpy')
    # hessian_f = sympy.lambdify((x1, x2, x3, x4), hessian_f, 'numpy')

    return sympy.lambdify((x1, x2, x3, x4), f, 'numpy'), grad_f, hessian_f


def wolfe_line_search(f, grad_f, x, d, alpha_init=1.0, c1=0.1, c2=0.5):
    # f: 目标函数
    # grad_f: 目标函数梯度
    # x: 当前点
    # d: 搜索方向
    # alpha_init: 步长初始值
    # c1: Wolfe条件中的常数
    # c2: Wolfe条件中的常数

    # 为了防止出现alpha_init过大，导致搜索时越过了极小点，我们需要将alpha_init进行限制
    alpha_max = 1.0e10
    alpha_min = 0
    alpha = min(alpha_init, alpha_max)

    phi_k = f(*x)
    phi_k_grad = np.array([fxi(*x) for fxi in grad_f])
    phi_k_1 = f(*(x + alpha * d))
    phi_k_1_grad = np.array([fxi(*(x + alpha * d)) for fxi in grad_f])

    while phi_k - phi_k_1 < - c1 * alpha * np.dot(phi_k_grad.T, d)\
            or np.dot(phi_k_1_grad.T, d) < c2 * np.dot(phi_k_grad.T, d):
        if np.dot(phi_k_grad.T, d) >= 0:
            return alpha

        # 不满足条件1
        if phi_k - phi_k_1 < - c1 * alpha * np.dot(phi_k_grad.T, d):
            alpha_max = alpha
            alpha = (alpha + alpha_min) / 2
        # 满足条件1 不满足条件2
        elif np.dot(phi_k_1_grad.T, d) < c2 * np.dot(phi_k_grad.T, d):
            alpha_min = alpha
            alpha = min(2 * alpha, (alpha + alpha_max) / 2)

        phi_k_1 = f(*(x + alpha * d))
        phi_k_1_grad = np.array([fxi(*(x + alpha * d)) for fxi in grad_f])

    return alpha


# def test_wolfe_line_search():
#     # 由于改成sympy库所以这个测试函数已经没有用了
#     def f(x):
#         return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) **2
#
#     def grad_f(x):
#         return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
#
#     x = np.array([0, 0])
#     d = np.array([1, 0])
#     alpha = wolfe_line_search(f, grad_f, x, d)
#
#     print("alpha: ", alpha)


def steepest_descent(f, grad_f, x0, x_opt, max_iter=1000, eps=1e-4):
    """
    最速下降法优化函数
    :param f: 目标函数
    :param grad_f: 目标函数的梯度函数
    :param x0: 初始点
    :param x_opt: 最优解
    :param max_iter: 最大迭代次数
    :param eps: 收敛精度
    :return: 最优解和最优解对应的函数值
    """
    x = x0.copy()
    g = np.array([fxi(*x) for fxi in grad_f])

    for k in range(max_iter):
        d = -g
        a = wolfe_line_search(f, grad_f, x, d)
        x_new = x + a * d
        g_new = np.array([fxi(*x_new) for fxi in grad_f])
        # print("iter:{}".format(k))

        if np.linalg.norm(x_opt - x_new) < eps:  # 梯度的模小于阈值时停止
            print('iteration count: ', k)
            x = x_new
            break
        x = x_new
        g = g_new
    return x, k


def damped_newton(f, grad_f, hess_f, x0, x_opt, max_iter=1000, eps=1e-4):
    """
    使用阻尼牛顿法来求解无约束优化问题

    :param f: 目标函数
    :param grad_f: 目标函数的梯度函数
    :param hess_f: 目标函数的海森矩阵函数
    :param x0: 初始点
    :param x_opt: 最优点
    :param max_iter: 最大迭代次数
    :param eps: 控制算法收敛精度的参数
    :return: 优化结果x
    """
    x = x0.copy()
    for k in range(max_iter):
        g = np.array([fxi(*x) for fxi in grad_f])
        H = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                H[i][j] = hess_f[i][j](*x)
        d = -np.linalg.solve(H, g)
        a = wolfe_line_search(f, grad_f, x, d)
        x_new = x + a * d
        if np.linalg.norm(x_new - x_opt) < eps:  # 梯度的模小于阈值时停止
            x = x_new
            break
        x = x_new
    return x, k


def DFP(f, grad_f, x0, x_opt, eps=1e-4, max_iter=1000):
    n = len(x0)
    A = np.eye(n)  # 初始化A
    x = x0
    g = np.array([fxi(*x) for fxi in grad_f])
    k = 0

    while np.linalg.norm(x - x_opt) > eps and k < max_iter:
        d = -np.dot(A, g)  # 计算搜索方向
        alpha = wolfe_line_search(f, grad_f, x, d)
        x_new = x + alpha * d  # 更新x
        g_new = np.array([fxi(*x_new) for fxi in grad_f])
        s = x_new - x  # 计算s和y
        y = g_new - g
        A = A + np.outer(s, s) / np.dot(s, y) - np.dot(np.dot(A, np.outer(y, y)), A) / np.dot(np.dot(y, A), y)  # 更新A
        x = x_new
        g = g_new
        k += 1

    return x, k


def FR(f, grad_f, x0, x_opt, max_iter=1000, eps=1e-6):
    n = len(x0)
    x = x0
    g = np.array([fxi(*x) for fxi in grad_f])
    H = np.eye(n)
    i = 0
    for i in range(max_iter):
        p = -np.dot(H, g)
        alpha = wolfe_line_search(f, grad_f, x, p)
        x_new = x + alpha * p
        g_new = np.array([fxi(*x_new) for fxi in grad_f])
        y = g_new - g
        if np.linalg.norm(x_new - x_opt) < eps:
            x = x_new
            break
        s = alpha * p
        Hy = np.dot(H, y)
        H += np.outer(s, s) / np.dot(y, s) - np.outer(Hy, Hy) / np.dot(Hy, y)
        x = x_new
        g = g_new
    return x, i


if __name__ == '__main__':

    f, grad_f, hessian_f = test_function()
    x0 = np.array([3, -1, 0, 1])
    x_opt = np.array([0, 0, 0, 0])

    print('Set the maximum number of iterations: 1000')
    print('Set the calculation accuracy: 10e-4')

    print('*********************steepest descent*********************')
    x, k = steepest_descent(f, grad_f, x0, x_opt)
    print('Number of iterations: ', k)
    print('x: ', x)
    print('Minimum value:', f(*x))

    print('*********************damped newton*********************')
    x, k = damped_newton(f, grad_f, hessian_f, x0, x_opt)
    print('Number of iterations: ', k)
    print('x: ', x)
    print('Minimum value:', f(*x))

    print('*********************DFP*********************')
    x, k = DFP(f, grad_f, x0, x_opt)
    print('Number of iterations: ', k)
    print('x: ', x)
    print('Minimum value:', f(*x))

    print('*********************FR*********************')
    x, k = FR(f, grad_f, x0, x_opt)
    print('Number of iterations: ', k)
    print('x: ', x)
    print('Minimum value:', f(*x))