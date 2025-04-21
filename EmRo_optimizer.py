import numpy as np
import random
import math

def error_rate(actual, pred):
    acc = np.sum(actual != pred) / actual.shape[0]
    error = 1 - acc
    return error

def fun(X):
    output = sum(np.square(X)) + random.random()
    return output

# Calculate fitness values for each population.
def CalculateFitness(X):
    fitness = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        fitness[i] = error_rate(X[i, :], np.load("Files/Labels.npy"))
        # fitness[i] =fitness[i]-np.random(1,2)
    return fitness

# Sort fitness values and corresponding positions.
def SortFitness(Fit):
    sorted_index = np.argsort(Fit)
    sorted_fitness = Fit[sorted_index]
    return sorted_fitness, sorted_index

# Sort the position according to fitness.
def SortPosition(X, index):
    return X[index, :]

# Boundary detection function.
def BorderCheck(X, lb, ub):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] < lb:
                X[i, j] = ub
            elif X[i, j] > ub:
                X[i, j] = lb
    return X


def MRFO(X, y, nPop, Dim, Max_iter, lb, ub):
    Dim = X.shape[1]
    X = np.random.uniform(lb, ub, size=(nPop, Dim))

    fitness = CalculateFitness(X)

    GbestScore = np.max(fitness)
    GbestPosition = X[np.argmin(fitness), :]

    Curve = np.zeros(Max_iter)

    for it in range(Max_iter):
        coef = it / Max_iter
        Fc = 2 - it * (2 / Max_iter)
        r5 = 0.5

        Xnew = np.zeros(X.shape)
        Xnew[0, :] = GbestPosition + np.random.random(Dim) * (GbestPosition - X[0, :])

        for i in range(nPop - 1):
            rr = np.random.random()
            if rr < r5:
                beta = 2 * math.exp(rr * ((Max_iter - it + 1) / Max_iter)) * math.sin(2 * math.pi * rr)
                if coef > np.random.random():
                    Xnew[i + 1, :] = GbestPosition + np.random.random(Dim) * (X[i, :] - X[i + 1, :]) + beta * (
                            GbestPosition - X[i + 1, :])
                else:
                    IndivRand = np.random.uniform(lb, ub, size=(1, Dim))
                    Xnew[i + 1, :] = IndivRand + np.random.random(Dim) * (X[i, :] - X[i + 1, :]) + beta * (
                            IndivRand - X[i + 1, :])
            else:
                randomlist2 = np.random.random(Dim)
                randomlist3 = -np.log(np.random.random(Dim))
                r1_soa = np.random.random()
                r2_soa = np.random.random()
                A1 = 2 * Fc * r1_soa - Fc
                C1 = 2 * r2_soa
                b = 1
                ll = (Fc - 1) * np.random.random() + 1
                Alpha = 2 * randomlist2 * np.sqrt(randomlist3)
                Xnew[i + 1, :] = X[i + 1, :] + np.random.random(Dim) * (X[i, :] - X[i + 1, :]) + Alpha * (
                        GbestPosition - X[i + 1, :])

        Xnew = BorderCheck(Xnew, lb, ub)
        new_fitness = CalculateFitness(Xnew)

        for i1 in range(nPop):
            if new_fitness[i1] < fitness[i1]:
                fitness[i1] = new_fitness[i1]
                X[i1, :] = Xnew[i1, :]

        best_index = np.argmax(fitness)
        # print('best_index:',best_index)
        # print('fitness:',fitness)
        if fitness[best_index] <= GbestScore:
            GbestScore = fitness[best_index]
            GbestPosition = X[best_index, :]

        Curve[it] = GbestScore

        # print('Iteration:', it)
        # print('Best fitness:', Curve[it])

    return GbestPosition, Curve



def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)

def generate_tent_map_array(shape, x0, r):
    rows, cols = shape
    array = np.zeros(shape)
    for i in range(rows):
        xn = x0  # Update initial value for each row
        for j in range(cols):
            xn = tent_map(xn, r)
            xn = max(0, min(1, xn))  # Clip the value within [0, 1]
            array[i, j] = xn
    return array
#%%
# shape = (12000, 100)  # Shape of the array
# x0 = 0.1  # Initial value
# r = 2.5   # Parameter value

# chaotic_array = generate_tent_map_array(shape, x0, r)
# popltion = np.array(chaotic_array)

# def ftr_selection():
#     # Set the parameters
#     nPop = 12000
#     Dim = 100
#     Max_iter = 15
#     lb = -1
#     ub = 1
    
#     # Generate initial population
#     X = np.random.uniform(lb, ub, size=(nPop, Dim))
#     popltion = np.array(chaotic_array)
#     y = np.load("Files/Labels.npy")
#     data = np.load("Files/Features.npy")
#     # Run MRFO feature selection
#     GbestPosition, Curve = MRFO(popltion, y, nPop, Dim, Max_iter, lb, ub)
    
#     # Selected feature subset
#     selected_features = GbestPosition > 0  
#     return selected_features
# sel_ftr = ftr_selection()
# selected_data = data[:, sel_ftr]
