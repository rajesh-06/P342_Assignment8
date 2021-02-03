# Creating a zero matrix of order m*n
def zeromatrix(m, n):
    p = [[0 for i in range(n)] for j in range(m)]
    return (p)


# Creating a identity matrix of m*m
def identity_mat(m):
    p = zeromatrix(m, m)
    for i in range(m):
        p[i][i] = 1
    return (p)


def mat_vec_mult(A, B):
    n = len(B)
    if len(A[0]) == n:
        p = [0 for i in range(n)]
        for i in range(n):
            for j in range(n):
                p[i] = p[i] + (A[i][j] * B[j])
        return (p)
    else:
        print('This combination is not suitable for multiplication')


# matrix multiplication
def mat_mult(a, b):
    if len(a[0]) == len(b):
        p = zeromatrix(len(a), len(b[0]))
        for i in range(len(a)):
            for j in range(len(b[0])):
                for x in range(len(b)):
                    p[i][j] += (a[i][x] * b[x][j])
        return (p)
    else:
        print('The matrix combination is not suitable for multiplication')


# Partial pivoting
def par_pivot(A, B):
    n = len(A)
    for r in range(n):
        if A[r][r] == 0:
            for r1 in range(r + 1, n):
                if abs(A[r1][r]) > A[r][r] and A[r][r] == 0:
                    (A[r], A[r1]) = (A[r1], A[r])
                    (B[r], B[r1]) = (B[r1], B[r])
                else:
                    continue
            else:
                continue


# Gauss-Jordan elimination
def gauss(A, B):
    m = len(A)
    n = len(A[0])
    for r in range(m):
        par_pivot(A, B)
        pivot = A[r][r]
        for c in range(r, n):
            A[r][c] = A[r][c] / pivot
        B[r] = B[r] / pivot
        for r1 in range(m):
            if r1 == r or A[r1][r] == 0:
                continue
            else:
                factor = A[r1][r]
                for c in range(r, n):
                    A[r1][c] = A[r1][c] - A[r][c] * factor
                B[r1] = B[r1] - B[r] * factor


# LU decomposition of a matrix
def lu_decompose(A, B):
    par_pivot(A, B)
    n = len(A)
    # To store in one matrix both L and U in matrix a
    try:
        import copy
        a = copy.deepcopy(A)
        for j in range(n):
            for i in range(n):
                factor = 0

                # for U(upper A) matrix
                if i <= j:
                    for k in range(i):
                        factor += a[i][k] * a[k][j]
                    a[i][j] = A[i][j] - factor
                # for L(lower) matrix
                else:
                    for k in range(j):
                        factor += a[i][k] * a[k][j]
                    a[i][j] = 1 / a[j][j] * (A[i][j] - factor)
    except ZeroDivisionError:
        print('LU decomposition is not possible.')

    return (a, B)


# for LUx=B
def lux(a, B):
    n = len(B)
    det = 1
    for i in range(n):
        det *= a[i][i]
    if len(a) == n and det != 0:
        print
        y = [0 for i in range(4)]
        x = [0 for i in range(4)]

        # forward substitution i.e., Ly=B
        for i in range(4):
            factor = 0
            for j in range(i):
                factor += a[i][j] * y[j]
            y[i] = B[i] - factor
        # Backward substitution, i.e. Ux=y
        for i in range(3, -1, -1):
            factor = 0
            for j in range(i + 1, 4, 1):
                factor += (a[i][j] * x[j])
            x[i] = 1 / a[i][i] * (y[i] - factor)
    return (x)


# for bracketing
def bracket(f, a, b):
    if f(a) == 0:
        print(a, 'is the root of the equation.')
    elif f(b) == 0:
        print(b, 'is the root of the equation.')
    else:
        while f(a) * f(b) > 0:
            if abs(f(a)) < abs(f(b)):
                a = a - 1.5 * (b - a)
            elif abs(f(a)) > abs(f(b)):
                b = b + 1.5 * (b - a)
    return a, b


# for finding a root using bisection method
def bisection(f, a, b):
    k = 0
    err = []
    print('SR.No.  Absolute error ')
    while abs(b - a) > 10 ** (-6) and k < 200:
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
            k += 1
        else:
            a = c
            k += 1
        err.append(c)
    n = len(err)
    arr = [0 for i in range(n - 1)]
    for i in range(n - 1):
        arr[i] = abs(err[i + 1] - err[i])
        print(i + 1, '     ', arr[i])

    return c, arr


# for finding a root using false position method
def fal_pos(f, a, b):
    k = 0
    err = []
    c = b - (((b - a) * f(b)) / (f(b) - f(a)))
    print('SR.No.  Absolute error ')
    while abs(f(c)) > 10 ** (-6) and k < 200:
        c = b - (((b - a) * f(b)) / (f(b) - f(a)))
        if f(a) * f(c) > 0:
            a = c
            k += 1
        else:
            b = c
            k += 1
        err.append(c)
    n = len(err)
    arr = [0 for i in range(n - 1)]
    for i in range(n - 1):
        arr[i] = abs(err[i + 1] - err[i])
        print(i + 1, '     ', arr[i])
    return c, arr


# for finding a root using Newton-raphson method
def newtraph(f, a):
    i = 0
    c = a
    err = []
    print('SR.No.  Absolute error ')
    while abs(f(c)) >= 10 ** (-10) and i < 200:
        c = a - f(a) / der1(f, a)
        i += 1
        a = c
        err.append(c)
        n = len(err)
    arr = [0 for i in range(n - 1)]
    for i in range(n - 1):
        arr[i] = abs(err[i + 1] - err[i])
        print(i + 1, '     ', arr[i])
    return c, arr


# 1st derivatives of a function
def der1(f, x):
    h = 10 ** (-3)
    f_ = (f(x + h) - f(x - h)) / (2 * h)
    return f_


# 2nd derivative of function
def der2(f, x):
    h = 10 ** (-3)
    f__ = (der(f, x + h) - der(f, x - h)) / (2 * h)
    return f__


# Value of p(x)
def poly(f, x):
    value = 0
    n = len(f)
    for i in range(n):
        value += f[i] * (x ** (n - 1 - i))
    return value


# 1st derivatives of p(x) at point x
def der1_poly(f, x):
    value = 0
    n = len(f)
    for i in range(n - 1):
        value += f[i] * (n - 1 - i) * (x ** (n - i - 2))
    return value


# 2nd derivative of p(x) at a point x
def der2_poly(f, x):
    value = 0
    n = len(f)
    for i in range(n - 2):
        value += f[i] * (n - 1 - i) * (n - 2 - i) * (x ** (n - i - 3))
    return value


def laguerre(f, x):
    h = 10 ** (-8)  # epsilon
    n = len(f) - 1  # degree of polynomial
    i = 0
    if abs(poly(f, x)) < h:  # checking
        return x
    else:
        while abs(poly(f, x)) > h and i < 100:
            g = der1_poly(f, x) / poly(f, x)
            h = g ** 2 - (der2_poly(f, x) / poly(f, x))
            d1 = g + (((n - 1) * (n * h - g ** 2)) ** 0.5)
            d2 = g - (((n - 1) * (n * h - g ** 2)) ** 0.5)
            # denominator should be larger
            if abs(d1) > abs(d2):
                a = n / d1
            else:
                a = n / d2
            x = x - a
            i += 1  # iteration number
        return x


# To find the root of polynomial using laguerre method
def root_poly(q2):
    deg = len(q2) - 1  # degree of polynomial
    # matrix to store root
    root = [0 for i in range(deg)]
    for i in range(deg):
        newp = []
        for j in range(deg + 1 - i):
            newp.append(q2[j])
        root[i] = laguerre(newp, 5)
        r = 0
        for j in range(deg - i):  # Resizing the polynomial after synthetic devision
            q2[j] += r * (root[i])
            r = q2[j]
    return root


import math as m


# integration of function 'f' with range [a,b], the range is divided with 'n' number of equal interval
def midpoint(f, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(n):
        x = a + (2 * i + 1) * (h / 2)
        sum += f(x) * h
    return (sum)


# To find n: 'f_max' is |f(x)''_max| in range [a,b] and 'err' is the maximum error to be compromised
def n_midpoint(f_max, a, b, err):
    n = ((b - a) ** 3 * f_max / (24 * err)) ** 0.5
    return m.ceil(n)


def trapezoidal(f, a, b, n):  # trapezoidal method
    h = (b - a) / n
    sum = (f(a) + f(b)) * (h / 2)
    for i in range(1, n):
        x = a + i * h
        sum += f(x) * h
    return (sum)


def n_trapezoidal(f_max, a, b, err):  # to find the N for a particular error for trapezoidal method
    n = ((b - a) ** 3 * f_max / (12 * err)) ** 0.5
    return m.ceil(n)


def simpson(f, a, b, n):  # simpson method
    h = (b - a) / n
    sum = (f(a) + f(b)) * (h / 3)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            sum += f(x) * (2 * h / 3)
        else:
            sum += f(x) * (4 * h / 3)
    return (sum)


def n_simpson(f_max, a, b, err):  # Calculating the value of N for an error "err"
    n = ((b - a) ** 5 * f_max / (180 * err)) ** 0.25
    if m.ceil(n) % 2 == 0:  # N should be smallest even number greater than n
        return m.ceil(n)
    else:
        return m.ceil(n) + 1


# Monte Carlo methods
def monte_carlo(f, a, b, n):
    sum = 0
    sum1 = 0
    import random as r
    for i in range(1, n + 1):
        x = a + (b - a) * r.random()  # creating random number between a to b
        sum += f(x) / n
        sum1 += (f(x) ** 2) / n
        error = (sum1 - (sum) ** 2) ** 0.5
    return ((b - a) * sum, error)  # returning both result and error


# Solution for differential equation using Euler's method of type: dy/dx=f(x) , y(x_0)=y
def euler(f, y, x, x_n, dx):  # taking parameter(dy/dx,y(x_0),x_0,x_max,dx)
    arrx = [x]  # storing the vaalue in list
    arry = [y]
    while x < x_n:  # loop upto x_max
        y += dx * f(y, x)
        x += dx
        arrx.append(x)  # updating list
        arry.append(y)
    return arry, arrx


def rk4(dvdx, y, v, x, dx, x_min, x_max):
    # this function is for solving 2nd order ODE with given y(x_0), dy/dx at x_0
    # x_min and x_max are boundary of x
    # dvdx = y''
    arrx = [x]  # creating list to store the values
    arry = [y]
    arrv = [v]
    while x > x_min:  # backward loop for boundary
        k1y = -dx * v
        k1v = -dx * dvdx(y, v, x)

        k2y = -dx * (v + 0.5 * k1v)
        k2v = -dx * dvdx(y + 0.5 * k1y, v + 0.5 * k1y, x - 0.5 * dx)

        k3y = -dx * (v + 0.5 * k2v)
        k3v = -dx * dvdx(y + 0.5 * k2y, v + 0.5 * k2y, x - 0.5 * dx)

        k4y = -dx * (v + 0.5 * k3v)
        k4v = -dx * dvdx(y + k3y, v + k3y, x - dx)

        y += (k2y + 2 * k2y + 2 * k3y + k4y) / 6
        v += (k2v + 2 * k2v + 2 * k3v + k4v) / 6
        x -= dx

        arry.append(y)  # appending the values in list
        arrv.append(v)
        arrx.append(x)
    x = arrx[0]  # assigning the initial values
    y = arry[0]
    v = arrv[0]
    # print(y,v,x)
    while x < x_max:  # forward loop
        k1y = dx * v
        k1v = dx * dvdx(y, v, x)

        k2y = dx * (v + 0.5 * k1v)
        k2v = dx * dvdx(y + 0.5 * k1y, v + 0.5 * k1y, x + 0.5 * dx)

        k3y = dx * (v + 0.5 * k2v)
        k3v = dx * dvdx(y + 0.5 * k2y, v + 0.5 * k2y, x + 0.5 * dx)

        k4y = dx * (v + 0.5 * k3v)
        k4v = dx * dvdx(y + k3y, v + k3y, x + dx)

        y += (k2y + 2 * k2y + 2 * k3y + k4y) / 6
        v += (k2v + 2 * k2v + 2 * k3v + k4v) / 6
        x += dx

        arry.append(y)  # appending
        arrv.append(v)
        arrx.append(x)
    return arrv, arry, arrx  # returning all the list


def shooting(dvdx, x1, y1, x2, y2, z, dx):
    # This function is for solving 2nd ODE with boundary values
    # uses rk4 method
    # y1 , y2 are boundary values at x1 and x2 respectively, dx is stepsize
    def yc(dvdx, y1, z, x1, dx, xl, xh):  # defining a function to return the last value of list_y in rk4
        v, y, x = rk4(dvdx, y1, z, x1, dx, xl, xh)
        return y[-1]

    beta = yc(dvdx, y1, z, x1, dx, x1, x2)
    beta_zl = beta
    beta_zh = beta
    if abs(beta - y2) < 0.001:  #
        v, y, x = rk4(dvdx, y1, z, x1, dx, x1, x2)
        return v, y, x  # returning all the list
    else:
        if beta > y2:  # if we get the upper bound
            zh = z
            while beta > y2:
                z -= 0.5  # decreasing to get the lower bound
                beta = yc(dvdx, y1, z, x1, dx, x1, x2)
            zl = z
            beta_zl = beta
            # langrange interpolation
            z = zl + (zh - zl) * (y2 - beta_zl) / (beta_zh - beta_zl)
            v, y, x = rk4(dvdx, y1, z, x1, dx, x1, x2)
            return v, y, x
        else:
            zl = z
            while beta < y2:  # if we get the lower bound
                z += 0.5  # incresing to get upper bound
                beta = yc(dvdx, y1, z, x1, dx, x1, x2)
            zh = z
            beta_zh = beta
            # langrange interpolation
            z = zl + (zh - zl) * (y2 - beta_zl) / (beta_zh - beta_zl)
            v, y, x = rk4(dvdx, y1, z, x1, dx, x1, x2)
        return v, y, x


import random as r


def rand_walk(length, step):
    x1 = 0
    y1 = 0
    xdata = [x1]  # x cordinate of the random walk
    ydata = [y1]  # y cordinate of the random walk
    for i in range(step - 1):
        theta = 2 * m.pi * r.random()
        x1 += length * m.cos(theta)
        y1 += length * m.sin(theta)
        xdata.append(x1)
        ydata.append(y1)
    return xdata, ydata  # returning the coordinte of each steps in a list


def creat_walk(length, step):
    x = []  #
    y = []
    r_x = []
    r_y = []
    for i in range(100):
        x1, y1 = rand_walk(length, step)
        x.append(x1)
        y.append(y1)
        r_x.append(x1[-1])
        r_y.append(y1[-1])
    return x, y, r_x, r_y
