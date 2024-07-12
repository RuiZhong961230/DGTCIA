# import packages
import os
import math
import scipy.stats as stats
import numpy as np
import warnings
from sklearn.svm import SVC
from numpy.random import seed
from numpy.random import randint
from sklearn.model_selection import cross_val_score
from os import path
import csv

# https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data

warnings.filterwarnings("ignore")
Xtrain = None
Ytrain = None

DimSize = 10

DuckPopSize = 30
DuckPop = np.zeros((DuckPopSize, DimSize))
FitDuck = np.zeros(DuckPopSize)
BestDuck = np.zeros(DimSize)
FitBestDuck = 0

FishPopSize = 20
FishPop = np.zeros((FishPopSize, DimSize))
FitFish = np.zeros(FishPopSize)
BestFish = np.zeros(DimSize)
FitBestFish = 0
Prey = np.zeros(DimSize)
CurrentBest = np.zeros(DimSize)
FitCurrentBest = 0
TotalPopSize = DuckPopSize + FishPopSize
LB = [-100] * DimSize  # the maximum value of the variable range
UB = [100] * DimSize  # the minimum value of the variable range
TrialRuns = 30  # the number of independent runs
MaxFEs = 2500
curIter = 0  # the current number of generations
MaxIter = math.ceil(MaxFEs / TotalPopSize)

FuncNum = 0

def open_csv(path):
    data = []
    with open(path) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            d = []
            for s in row:
                d.append(float(s))
            data.append(d)
    return np.array(data)


def Transfer(kk):
    new_kk = np.zeros(len(kk))
    for i in range(len(kk)):
        if kk[i] < 0:
            new_kk[i] = 0
        else:
            new_kk[i] = 1
    return new_kk


def fit_knn(kk):
    global Xtrain
    kk = Transfer(kk)
    k = 5
    if len(kk) == 0 or sum(kk) == 0:
        seed(1)
        kk = randint(0, 2, np.size(Xtrain, 1))
    pos = []
    for i in range(0, np.size(Xtrain, 1)):
        if kk[i] == 1:
            pos.append(i)

    model = SVC()
    X = Xtrain[:, pos]
    scores = cross_val_score(model, X, Ytrain, cv=5)
    cost = sum(scores) / k
    return cost * 100


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def Initialization():
    global DuckPop,FitDuck,FishPop,FitFish,BestDuck, BestFish, Prey, CurrentBest, FitCurrentBest,FitBestFish,FitBestDuck,CurrentBest,FitCurrentBest
    DuckPop = np.zeros((DuckPopSize, DimSize))
    FitDuck = np.zeros(DuckPopSize)
    BestDuck = np.zeros(DimSize)
    FishPop = np.zeros((FishPopSize, DimSize))
    FitFish = np.zeros(FishPopSize)
    BestFish = np.zeros(DimSize)
    Prey = np.zeros(DimSize)
    CurrentBest = np.zeros(DimSize)
    FitCurrentBest = 0
    # randomly generate individuals
    for i in range(DuckPopSize):
        for j in range(DimSize):
            DuckPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitDuck[i] = -fit_knn(DuckPop[i])
    for i in range(FishPopSize):
        for j in range(DimSize):
            FishPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitFish[i] = -fit_knn(FishPop[i])
    BestDuck = DuckPop[np.argmin(FitDuck)]
    FitBestDuck = np.min(FitDuck)

    BestFish = FishPop[np.argmin(FitFish)]
    FitBestFish = np.min(FitFish)
    if FitBestFish > FitBestDuck:
        CurrentBest = np.copy(BestDuck)
        FitCurrentBest = FitBestDuck
    else:
        CurrentBest = np.copy(BestFish)
        FitCurrentBest = FitBestFish


def Check(indi):
    global LB, UB, DimSize
    for j in range(DimSize):
        if indi[j] > UB[j] or indi[j] < LB[j]:
            indi[j] = np.random.uniform(LB[j], UB[j])
    return indi


def DGTCIA():
    global DuckPop, FitDuck, BestDuck, FitBestDuck, FishPop, FitFish, BestFish, FitBestFish, Prey, CurrentBest, FitCurrentBest
    # find the best duck and fish
    BestDuck = DuckPop[np.argmin(FitDuck)]
    FitBestDuck = np.min(FitDuck)
    BestFish = FishPop[np.argmin(FitFish)]
    FitBestFish = np.min(FitFish)
    Off = np.zeros(DimSize)
    idx_duck = np.argmin(FitDuck)
    idx_fish = np.argmin(FitFish)

    A = (1 - curIter / MaxIter)
    for i in range(DuckPopSize):
        if i == idx_duck:
            target = []
            for j in range(DimSize):
                if np.random.rand() < np.random.normal(0.01, 0.01):
                    target.append(j)
            F = np.random.normal(0.5, 0.3)
            candi = list(range(0, DuckPopSize))
            r1, r2 = np.random.choice(candi, 2, replace=False)
            for j in range(DimSize):
                if j in target:
                    Off[j] = BestDuck[j]
                else:
                    if np.random.rand() < 0.05:
                        r = np.random.randint(0, DuckPopSize)
                        Off[j] = BestDuck[j] + F * (DuckPop[r1][j] - DuckPop[r][j])
                    else:
                        Off[j] = BestDuck[j] + F * (DuckPop[r1][j] - DuckPop[r2][j])
        else:
            idx = np.random.randint(0, FishPopSize)
            RandFit = FitFish[idx]
            diversity = np.mean(np.std(DuckPop, axis=0))
            if FitDuck[i] < RandFit:
                Off = DuckPop[i] + np.random.random() * (CurrentBest - DuckPop[i]) * np.sin(2 * np.pi * np.random.random()) * A * diversity
            else:

                for j in range(DimSize):
                    if np.random.random() < 0.5:
                        Off[j] = DuckPop[i][j] + np.random.normal() * A * diversity
                    else:
                        Off[j] = FishPop[idx][j] + np.random.normal() * A * diversity

        Off = Check(Off)
        FitOff = -fit_knn(Off)
        if FitOff < FitDuck[i]:
            DuckPop[i] = np.copy(Off)
            FitDuck[i] = FitOff
            if FitOff < FitBestDuck:
                BestDuck = np.copy(Off)
                FitBestDuck = FitOff
                if FitOff < FitCurrentBest:
                    CurrentBest = np.copy(Off)
                    FitCurrentBest = FitOff
    for i in range(FishPopSize):
        if i == idx_fish:
            target = []
            for j in range(DimSize):
                if np.random.rand() < np.random.normal(0.01, 0.01):
                    target.append(j)
            F = np.random.normal(0.5, 0.3)
            candi = list(range(0, FishPopSize))
            r1, r2 = np.random.choice(candi, 2, replace=False)
            for j in range(DimSize):
                if j in target:
                    Off[j] = BestFish[j]
                else:
                    if np.random.rand() < 0.05:
                        r = np.random.randint(0, FishPopSize)
                        Off[j] = BestFish[j] + F * (FishPop[r1][j] - FishPop[r][j])
                    else:
                        Off[j] = BestFish[j] + F * (FishPop[r1][j] - FishPop[r2][j])
        else:
            Off = BestFish - np.random.uniform(-0.5, 0.5) * (BestDuck - FishPop[i]) + np.random.uniform(-0.5, 0.5) * (DuckPop[np.random.randint(0, DuckPopSize)] - FishPop[i])

        Off = Check(Off)
        FitOff = -fit_knn(Off)
        if FitOff < FitFish[i]:
            FishPop[i] = np.copy(Off)
            FitFish[i] = FitOff
            if FitOff < FitBestFish:
                BestFish = np.copy(Off)
                FitBestFish = FitOff
                if FitOff < FitCurrentBest:
                    CurrentBest = np.copy(Off)
                    FitCurrentBest = FitOff


def RunDGTCIA():
    global curIter, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    All_Best_Scale = []
    for i in range(TrialRuns):
        np.random.seed(2000 + 88 * i)
        Best_list = []
        curIter = 0
        Initialization()
        curIter = 1
        Best_list.append(FitCurrentBest)
        while curIter < MaxIter:
            DGTCIA()
            curIter += 1
            Best_list.append(FitCurrentBest)
        All_Trial_Best.append(np.abs(Best_list))
        All_Best_Scale.append(sum(Transfer(CurrentBest)))
    np.savetxt('./DGTCIA_Data/Acc/data.csv', All_Trial_Best, delimiter=",")
    np.savetxt('./DGTCIA_Data/Scale/data.csv', All_Best_Scale, delimiter=",")


def main():
    global FuncNum, DimSize, MaxFEs, MaxIter, SuiteName, LB, UB, Xtrain, Ytrain
    dataset = open_csv(path.dirname(path.abspath(__file__)) + "\\data.csv")
    Xtrain = stats.zscore(np.asarray(dataset[:, 1:31]))
    Ytrain = np.asarray(dataset[:, 0])
    DimSize = len(Xtrain[0])
    LB = [-100] * DimSize
    UB = [100] * DimSize
    RunDGTCIA()


if __name__ == "__main__":
    if os.path.exists('./DGTCIA_Data/Acc') == False:
        os.makedirs('./DGTCIA_Data/Acc')
    if os.path.exists('./DGTCIA_Data/Scale') == False:
        os.makedirs('./DGTCIA_Data/Scale')
    main()


