# import packages
import os
import math
from opfunu.cec_based.cec2022 import *
import numpy as np


DimSize = 10

DuckPopSize = 60
DuckPop = np.zeros((DuckPopSize, DimSize))
FitDuck = np.zeros(DuckPopSize)
BestDuck = np.zeros(DimSize)
FitBestDuck = 0

FishPopSize = 40
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
MaxFEs = 1000 * DimSize  # the maximum number of fitness evaluations


Fun_num = 1  # the serial number of benchmark function
curIter = 0  # the current number of generations
MaxIter = math.ceil(MaxFEs / TotalPopSize)


FuncNum = 0
SuiteName = "CEC2022"


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def Initialization(func):
    global DuckPop, FitDuck, FishPop, FitFish, BestDuck, BestFish, Prey, CurrentBest, FitCurrentBest, FitBestFish, FitBestDuck, CurrentBest, FitCurrentBest
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
        FitDuck[i] = func(DuckPop[i])
    for i in range(FishPopSize):
        for j in range(DimSize):
            FishPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitFish[i] = func(FishPop[i])
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


def DGTCIA(func):
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
        FitOff = func(Off)
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
        FitOff = func(Off)
        if FitOff < FitFish[i]:
            FishPop[i] = np.copy(Off)
            FitFish[i] = FitOff
            if FitOff < FitBestFish:
                BestFish = np.copy(Off)
                FitBestFish = FitOff
                if FitOff < FitCurrentBest:
                    CurrentBest = np.copy(Off)
                    FitCurrentBest = FitOff


def Run(func):
    global curIter, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        np.random.seed(2000 + 88 * i)
        Best_list = []
        curIter = 0
        Initialization(func)
        curIter = 1
        Best_list.append(FitCurrentBest)
        while curIter < MaxIter:
            DGTCIA(func)
            curIter += 1
            Best_list.append(FitCurrentBest)
        All_Trial_Best.append(Best_list)
    np.savetxt("./DGTCIA_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, SuiteName, LB, UB
    DimSize = dim
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / TotalPopSize)
    LB = [-100] * dim
    UB = [100] * dim
    CEC2022 = [F12022(Dim), F22022(Dim), F32022(Dim), F42022(Dim), F52022(Dim), F62022(Dim),
               F72022(Dim), F82022(Dim), F92022(Dim), F102022(Dim), F112022(Dim), F122022(Dim)]

    for i in range(len(CEC2022)):
        FuncNum = i + 1
        Run(CEC2022[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('DGTCIA_Data/CEC2022') == False:
        os.makedirs('DGTCIA_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)


