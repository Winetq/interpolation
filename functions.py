from math import sqrt
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg


def load_data(file_name):
    data = pd.read_csv(file_name)
    return data['Dystans (m)'], data['Wysokość (m)'], len(data)


def load_data_with_step(file_name, step):
    data = pd.read_csv(file_name)
    x = []
    y = []
    x.append(data['Dystans (m)'][0])
    y.append(data['Wysokość (m)'][0])
    for i in range(step, len(data) - 1, step):
        x.append(data['Dystans (m)'][i])
        y.append(data['Wysokość (m)'][i])
    x.append(data['Dystans (m)'][len(data) - 1])
    y.append(data['Wysokość (m)'][len(data) - 1])
    return x, y


def load_data_with_n(file_name, n):
    data = pd.read_csv(file_name)
    x = []
    y = []
    x.append(data['Dystans (m)'][0])
    y.append(data['Wysokość (m)'][0])
    n = 50 if n > len(data) else n
    for _ in range(1, n - 1):
        i = randint(1, len(data) - 2)
        x.append(data['Dystans (m)'][i])
        y.append(data['Wysokość (m)'][i])
    x.append(data['Dystans (m)'][len(data) - 1])
    y.append(data['Wysokość (m)'][len(data) - 1])

    tuples = []
    for i in range(len(x)):
        tuples.append((x[i], y[i]))

    tuples.sort(key=lambda temp: temp[0])

    x = [i[0] for i in tuples]
    y = [i[1] for i in tuples]

    return x, y


def comparison(x, y, x1, y1, x2, y2, title):
    plt.figure(figsize=(14, 6))
    plt.plot(x1, y1, label="funkcja oryginalna")
    plt.plot(x2, y2, label="funkcja interpolowana")
    plt.plot(x, y, '*', label="węzły interpolacji")
    plt.xlabel('dystans [m]')
    plt.ylabel('wysokość [m]')
    plt.title(title)
    plt.legend()
    plt.show()


def Lagrange_algorithm(x_original, x, y):
    x_Lagrange = []
    y_Lagrange = []
    for i in range(len(x_original)):
        f = 0
        for j in range(len(x)):
            a = 1
            for k in range(len(y)):
                if j != k:
                    a *= (x_original[i] - x[k])/(x[j] - x[k])
            f += a * y[j]
        x_Lagrange.append(x_original[i])
        y_Lagrange.append(f)

    return x_Lagrange, y_Lagrange


def spline_algorithm(x_original, x1, y1):
    n = len(x1) - 1
    # A * x = b
    A = np.zeros(shape=(4 * n, 4 * n))
    b = np.zeros(shape=(4 * n, 1))

    for i in range(0, n):
        A[i][4 * i + 0] = x1[i] ** 3
        A[i][4 * i + 1] = x1[i] ** 2
        A[i][4 * i + 2] = x1[i]
        A[i][4 * i + 3] = 1
        b[i] = y1[i]

        A[n + i][4 * i + 0] = x1[i + 1] ** 3
        A[n + i][4 * i + 1] = x1[i + 1] ** 2
        A[n + i][4 * i + 2] = x1[i + 1]
        A[n + i][4 * i + 3] = 1
        b[n + i] = y1[i + 1]

        if i == 0:
            continue

        A[2 * n + (i - 1)][4 * (i - 1) + 0] = 3 * x1[i] ** 2
        A[2 * n + (i - 1)][4 * (i - 1) + 1] = 2 * x1[i]
        A[2 * n + (i - 1)][4 * (i - 1) + 2] = 1
        A[2 * n + (i - 1)][4 * (i - 1) + 4] = -3 * x1[i] ** 2
        A[2 * n + (i - 1)][4 * (i - 1) + 1 + 4] = -2 * x1[i]
        A[2 * n + (i - 1)][4 * (i - 1) + 2 + 4] = -1
        b[2 * n + (i - 1)] = 0

        A[3 * n + (i - 1)][4 * (i - 1) + 0] = 6 * x1[i]
        A[3 * n + (i - 1)][4 * (i - 1) + 1] = 2
        A[3 * n + (i - 1)][4 * (i - 1) + 0 + 4] = -6 * x1[i]
        A[3 * n + (i - 1)][4 * (i - 1) + 1 + 4] = -2
        b[3 * n + (i - 1)] = 0

    A[3 * n - 1 + 0][0 + 0] += 6 * x1[0]
    A[3 * n - 1 + 0][0 + 1] += 2
    b[3 * n - 1 + 0] += 0

    A[3 * n + n - 1][4 * (n - 1) + 0] += 6 * x1[n]
    A[3 * n + n - 1][4 * (n - 1) + 1] += 2
    b[3 * n + n - 1] += 0

    LU, piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve((LU, piv), b)

    x_spline = []
    y_spline = []
    for i in range(len(x_original)):
        for j in range(n):
            if x1[j] <= x_original[i] <= x1[j + 1]:
                h = x_original[i]
                f = x[4*j+3] + x[4*j+2]*h + x[4*j+1]*h*h + x[4*j+0]*h*h*h
                x_spline.append(x_original[i])
                y_spline.append(f)
                break

    return x_spline, y_spline


def count_RMSD(y, y1):
    sigma = 0
    for i in range(len(y)):
        sigma += (y1[i] - y[i]) * (y1[i] - y[i])
    ans = sqrt(sigma/len(y))
    print("RMSD = " + str(ans))
