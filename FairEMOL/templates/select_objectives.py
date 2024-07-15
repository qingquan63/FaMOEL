####    Authors:    Handing Wang,  Xin Yao
####    Xidian University, China, and University of Birmingham, UK
####    EMAIL:      wanghanding.patch@gmail.com, X.Yao@cs.bham.ac.uk
####    WEBSITE:    http://www.cs.bham.ac.uk/~xin/
####    DATE:       March 2015
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
import copy

# Handing Wang,  Xin Yao, Objective Reduction Based on Nonlinear Correlation Information Entropy, Soft Computing, Accepted.

# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------
# You can run Two_Arch2 by calling function [ Sr ] = Select_Ojbectives( obj)
# to obtain reducted objective subset. You can use this function in every
# generation of Pareto-based MOEAs to reduce the number of objectives.


import numpy as np


def Select_Ojbectives_cut(obj=None, R=None, throd=0.2):
    # Usage: [ Sr ] = Select_Ojbectives( obj )

    # Input:
    # obj           - objective values of the non-dominated set

    # Output:
    # Sr            - reducted objective subset(index)

    c = obj.shape[2 - 1]

    St = np.arange(0, c)
    Sr = []
    if R is None:
        R = RNCC(obj)
    R_res = copy.deepcopy(R)

    i = 0
    while len(St) != 0:

        t = R < 0
        s = np.sum(t, axis=0)
        sx = np.amax(s)
        index = np.argmax(s)
        if sx == 0:
            t = sum(R)
            sx = np.amax(t)
            index = np.argmax(t)
        i = i + 1
        Sr.append(St[index])
        t = R[np.concatenate((np.arange(0, index), np.arange(index + 1, len(R))), axis=0), index]
        St = np.delete(St, index)
        if len(St) == 0:
            break
        R = np.delete(R, index, 0)
        R = np.delete(R, index, 1)
        if np.amax(t) > 0:
            I1 = list(np.where(t >= 0))
            n = len(I1[0])
            tt = t[tuple(I1)]
            ind = np.where(tt > throd)
            # ind = np.where(tt > 0.95*np.max(tt))
            index2 = I1[0][ind[0]]
        else:
            index2 = []
        St = np.delete(St, index2)
        R = np.delete(R, index2, 0)
        R = np.delete(R, index2, 1)
    return Sr, R_res


def RNCC(r=None):
    # Usage: [ R ] = RNCC(r)

    # Input:
    # r             - dataset

    # Output:
    # R             - NCIE matrix

    c = r.shape[2 - 1]
    R = np.eye(c)
    for i in np.arange(0, c).reshape(-1):
        for j in np.arange(i + 1, c).reshape(-1):
            R[i, j] = NCC1(r[:, i], r[:, j])
            R[j, i] = R[i, j]

    return R


def NCC1(x=None, y=None):
    # Usage: [ ncc ] = NCC1( x,y)

    # Input:
    # x              - variable
    # y              - variable

    # Output:
    # ncc            - NCIE of x and y

    n = x.shape[1 - 1]
    b = int(np.around(n ** 0.5))
    if np.amax(x) != np.amin(x) and np.amax(y) != np.amin(y):
        detax = (np.amax(x) - np.amin(x) + 1e-05 * (np.amax(x) - np.amin(x))) / b
        detay = (np.amax(y) - np.amin(y) + 1e-05 * (np.amax(y) - np.amin(y))) / b
        if detax != 0 and detay != 0:
            p = np.zeros([b+1, b+1])
            x1 = np.ceil((x - np.amin(x) + 5e-06 * (np.amax(x) - np.amin(x))) / detax)
            y1 = np.ceil((y - np.amin(y) + 5e-06 * (np.amax(y) - np.amin(y))) / detay)
            x1[np.where(x1 <= 0)] = 1
            y1[np.where(y1 <= 0)] = 1
            x1[np.where(x1 > b)] = b
            y1[np.where(y1 > b)] = b
            x1 = x1.astype('int32')
            y1 = y1.astype('int32')
            for i in np.arange(0, n).reshape(-1):
                p[x1[i], y1[i]] = p[x1[i], y1[i]] + 1 / n
            ncc = 0
            for i in np.arange(1, b+1).reshape(-1):
                for j in np.arange(1, b+1).reshape(-1):
                    if p[i, j] != 0:
                        ncc = ncc + (p[i, j]) * np.log(p[i, j]) / np.log(b)
            for i in np.arange(1, b + 1).reshape(-1):
                if sum(p[i, :]) != 0:
                    ncc = ncc - (sum(p[i, :])) * np.log(sum(p[i, :])) / np.log(b)
            for i in np.arange(1, b + 1).reshape(-1):
                if sum(p[:, i]) != 0:
                    ncc = ncc - (sum(p[:, i])) * np.log(sum(p[:, i])) / np.log(b)
        else:
            ncc = 0
    else:
        ncc = 0

    cor = np.cov(x, y)
    if cor[0, 1] < 0:
        ncc = 0 - ncc

    return ncc


if __name__ == '__main__':
    obj = np.loadtxt('F:\Download\ObjectiveReduction-master\obj.txt')
    res = Select_Ojbectives_cut(obj)
    print(res)
    print('over')
