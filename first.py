import numpy as np
import sympy


def steepest_descent(f, grad_f, x0, max_iter=100, eps=1e-6):
    """
    最速下降法优化函数
    :param f: 目标函数
    :param grad_f: 目标函数的梯度函数
    :param x0: 初始点
    :param max_iter: 最大迭代次数
    :param eps: 收敛精度
    :return: 最优解和最优解对应的函数值
    """
    x = x0.copy()
    g = grad_f(x)

    for k in range(max_iter):
        d = -g
        a = line_search(f, grad_f, x, d)
        x_new = x + a * d
        g_new = grad_f(x_new)
        if np.linalg.norm(g_new) < eps:  # 梯度的模小于阈值时停止
            print('iteration count: ', k)
            x = x_new
            break
        x = x_new
        g = g_new
    return x


def damped_newton(f, grad_f, hess_f, x0, max_iter=100, eps=1e-6):
    """
    使用阻尼牛顿法来求解无约束优化问题

    :param f: 目标函数
    :param grad_f: 目标函数的梯度函数
    :param hess_f: 目标函数的海森矩阵函数
    :param x0: 初始点
    :param max_iter: 最大迭代次数
    :param eps: 控制算法收敛精度的参数
    :return: 优化结果x
    """
    x = x0.copy()
    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        d = -np.linalg.solve(H, g)
        a = line_search(f, grad_f, x, d)
        x_new = x + a * d
        g_new = grad_f(x_new)
        if np.linalg.norm(g_new) < eps:  # 梯度的模小于阈值时停止
            print('iteration count: ', k)
            x = x_new
            break
        x = x_new
    return x


def BFGS(f, grad_f, x0, max_iter=100, eps=1e-6):
    """
    BFGS算法实现函数，用于求解无约束优化问题的最优解。
    :param f: 目标函数，输入当前自变量x，返回函数值f和梯度g。
    :param grad_f: 目标函数的梯度函数，输入当前自变量x，返回梯度g。
    :param x0: 自变量的初始值。
    :param max_iter: 最大迭代次数。
    :param eps: 梯度的模的阈值，当梯度的模小于该值时，停止迭代。
    :return: 优化问题的最优解x。
    """
    n = len(x0)
    H = np.eye(n)
    x = x0.copy()
    g = grad_f(x)

    for k in range(max_iter):
        d = -np.dot(H, g)
        a = line_search(f, grad_f, x, d)
        x_new = x + a * d
        g_new = grad_f(x_new)
        y = g_new - g
        if np.linalg.norm(g_new) < eps:  # 梯度的模小于阈值时停止
            x = x_new
            break
        s = a * d
        rho = 1 / np.dot(y.T, s)
        H = np.dot((np.eye(n) - rho * np.outer(s, y)), np.dot(H, (np.eye(n) - rho * np.outer(y, s)))) + rho * np.outer(
            s, s)
        x = x_new
        g = g_new
    return x


def conjugate_gradient_method(f, grad_f, x0, max_iter=100, eps=1e-6):
    """
    共轭梯度法求解无约束非线性规划问题的极小点

    :param f: 输入函数，形如 f(x)
    :param grad_f: 函数梯度，形如 grad_f(x)
    :param x0: 初始解
    :param max_iter: 最大迭代次数
    :param eps: 停止迭代的阈值，当解的改变量小于阈值时停止
    :return: 近似最优解 x
    """
    x = x0.copy()
    g = grad_f(x)
    d = -g
    for k in range(max_iter):
        a = line_search(f, grad_f, x, d)
        x_new = x + a * d
        g_new = grad_f(x_new)
        if np.linalg.norm(g_new) < eps:
            print('iteration count: ', k)
            x = x_new
            break
        beta = np.dot(g_new.T, g_new) / np.dot(g.T, g)
        d_new = -g_new + beta * d
        x, g, d = x_new, g_new, d_new
    return x


def line_search(f, grad_f, x, d):
    a = 1.0
    rho = 0.5
    c = 1e-4
    phi0 = f(x)
    phi = f(x + a * d)
    phi_prime = np.dot(grad_f(x + a * d).T, d)
    while phi > phi0 + c * a * phi_prime:
        a = rho * a
        phi = f(x + a * d)
    return a


def golden_section_search(f, a, b, eps):
    """
    黄金分割法进行精确搜索
    :param f: 目标函数
    :param a: 初始区间左端点
    :param b: 初始区间右端点
    :param eps: 停止准则，当区间长度小于eps时停止搜索
    :return: 区间内函数极小值点
    """
    rho = (3 - np.sqrt(5)) / 2  # 黄金分割常数
    x1 = a + rho * (b - a)
    x2 = b - rho * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    while (b - a) >= eps:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + rho * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - rho * (b - a)
            f2 = f(x2)
    return (a + b) / 2


def easy_alpha(x, d):
    a1 = sympy.symbols('a1')
    x1 = x + a1 * d
    f = 0.5 * x1.T @ G @ x1 + b.T @ x1
    df = sympy.diff(f, a1)
    a = float(sympy.solve(df, a1)[0])
    return a


if __name__ == '__main__':
    # 设置n 2ab
    n = 257
    x0 = np.zeros((n, 1))
    a = np.random.randint(10, size=(n, 1))
    global G, b
    G = a.dot(a.T) + np.random.randint(2) * np.eye(n)
    b = 0.5 * G.dot(np.ones((n, 1)))

    # 保证G为正定阵
    while np.any(np.linalg.eigvals(G) <= 0):
        G = a.dot(a.T) + np.random.randint(2) * np.eye(n)

    def test_function(x):
        return 0.5 * np.dot(x.T, np.dot(G, x)) + np.dot(b.T, x)

    # 测试函数的梯度
    def grad_test_function(x):
        return np.dot(G, x) + b

    # 测试函数的海森矩阵
    def hessian_test_function(x):
        return G

    print('*********************steepest descent*********************')
    x = steepest_descent(test_function, grad_test_function, x0)
    print('Initial point(first five value):', x0[:5, 0].T)
    print('Dimension:', n)
    print('Optimal solution(first five value):', x[:5, 0].T)
    print('Minimum value:', test_function(x))

    print('*********************damped newton*********************')
    x = damped_newton(test_function, grad_test_function, hessian_test_function, x0)
    print('Initial point(first five value):', x0[:5, 0].T)
    print('Dimension:', n)
    print('Optimal solution(first five value):', x[:5, 0].T)
    print('Minimum value:', test_function(x))

    print('*********************BFGS*********************')
    x = BFGS(test_function, grad_test_function, x0)
    print('Initial point(first five value):', x0[:5, 0].T)
    print('Dimension:', n)
    print('Optimal solution(first five value):', x[:5, 0].T)
    print('Minimum value:', test_function(x))

    print('*********************conjugate gradient method*********************')
    x = conjugate_gradient_method(test_function, grad_test_function, x0)
    print('Initial point(first five value):', x0[:5, 0].T)
    print('Dimension:', n)
    print('Optimal solution(first five value):', x[:5, 0].T)
    print('Minimum value:', test_function(x))
