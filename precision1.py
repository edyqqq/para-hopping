import numpy as np
import cv2
from decimal import Decimal, getcontext
import math
import matplotlib.pyplot as plt
from collections import Counter
import copy
import time

def hopped(k):
    h = 4
    amax = 3.81-0.0001
    amin = 3.53+0.0001
    bmax = 0.022
    bmin = 0.0001
    cmax = 0.015
    cmin = 0.0001
    new_a = Decimal(amax) - Decimal(k) * Decimal(amax - amin)
    new_b = Decimal(bmax) - Decimal(k) * Decimal(bmax - bmin)
    new_c = Decimal(cmax) - Decimal(k) * Decimal(cmax - cmin)
    new_k = Decimal(h) * Decimal(k) * Decimal(1 - k)

    return new_a, new_b, new_c, new_k

def get_cs(x1, y1, z1, a1, b1, c1, M, N):
    cs_1, cs_2, cs_3, k_list = [], [], [], []
    x, y, z, a, b, c, k = Decimal(x1), Decimal(y1), Decimal(z1), Decimal(a1), Decimal(b1), Decimal(c1), Decimal(0.01)
    cs_1.append(float(x))
    cs_2.append(float(y))
    cs_3.append(float(z))
    for i in range(M*N):
        x_new = a*x*(1-x) + b*(y**2)*x + c*(z**3)
        y_new = a*y*(1-y) + b*(z**2)*y + c*(x**3)
        z_new = a*z*(1-z) + b*(x**2)*z + c*(y**3)
        cs_1.append(float(x_new))
        cs_2.append(float(y_new))
        cs_3.append(float(z_new))
        k_list.append(float(k))
        a, b, c, k = hopped(k)
        x, y, z = x_new, y_new, z_new

    return np.array(cs_1), np.array(cs_2), np.array(cs_3), np.array(k_list)

def trajectory(A, B, C, D):
    b = dict(Counter(A.tolist()))
    print([key for key,value in b.items() if value > 1])
    print({key:value for key,value in b.items()if value > 1})
    b = dict(Counter(B.tolist()))
    print([key for key,value in b.items() if value > 1])
    print({key:value for key,value in b.items()if value > 1})
    textfile = open(f"./test result/hopped_256/value_pre{getcontext().prec}.txt", "w")
    for i in range(len(B)):
        textfile.write(f'{B[i]},{C[i]},{D[i]}' + "\n")
    textfile.close()
    textfile = open(f"./test result/hopped_256/para_pre{getcontext().prec}.txt", "w")
    for element in A.tolist():
        textfile.write(f'{element}' + "\n")
    textfile.close()

def row_rotate(h, cs_x, eta1, I):
    rr = cs_x[eta1: eta1 + h].tolist()
    clean_list(rr)
    rr_sort = copy.deepcopy(rr)
    rr_sort.sort()
    new_I = np.zeros((I.shape[0],I.shape[1]), dtype = np.uint8)
    i = 0
    for element in rr:
        new_I[rr_sort.index(element), :] = I[i, :]
        i += 1
    
    return new_I

def col_rotate(w, cs_y, eta3, I):
    cr = cs_y[eta3: eta3 + w].tolist()
    clean_list(cr)
    cr_sort = copy.deepcopy(cr)
    cr_sort.sort()
    new_I = np.zeros((I.shape[0],I.shape[1]), dtype = np.uint8)
    i = 0
    for element in cr:
        new_I[:, cr_sort.index(element)] = I[:, i]
        i += 1
    
    return new_I

def clean_list(L):
    min = 0
    for i in range(len(L)):
        if L[i] in L[:i]:
            while min in L[:i]:
                min += 1
            L[i] = min    
    
    return L

def XOR(I, cs_z, eta5):
    np.reshape(I, (1, I.shape[0]*I.shape[1]))
    for i in range(len(I)):
        I[i] = I[i] ^ int(cs_z[eta5 + i])
    return I

def main():
    start_time = time.time()
    # read img
    img = cv2.imread("./Lenna_256.bmp", cv2.IMREAD_GRAYSCALE)
    # set parameter
    getcontext().prec = 16          
    a_1 = 3.79
    b_1 = 0.0185
    c_1 = 0.0125
    x_1 = 0.235
    y_1 = 0.35
    z_1 = 0.735
    eta1, eta2, eta3, eta4, eta5, eta6 = 500, 100000, 600, 100000, 700, 100000
    m = img.shape[0]
    n = img.shape[1]
    # get chaotic seq1,2,3
    X, Y, Z, K = get_cs(x_1, y_1, z_1, a_1, b_1, c_1, 256, 256)

    # histogram equalize
    X_he = np.mod(np.floor(X * eta2), m)
    Y_he = np.mod(np.floor(Y * eta4), n)
    Z_he = np.mod(np.floor(Z * eta6), 256)

    # row rotate and column rotate(confusion)
    confuse_img = row_rotate(m, X_he, eta1, img)
    final_img = col_rotate(m, Y_he, eta3, confuse_img)

    # XOR operation(diffusion)
    en_img = XOR(final_img, Z_he, eta5)
    end_time = time.time()
    print((end_time-start_time))
    # cv2.imwrite('lena_256_en.png', en_img)
    cv2.imshow('img', en_img)
    cv2.waitKey()


if __name__ == '__main__':
    main()