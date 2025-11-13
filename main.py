# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 08:40:56 2025

@author: e2204673
"""
import math
import numpy as np
import pandas as pd

def read_data(path : str) -> pd.DataFrame :
    data = pd.read_csv(path, 
                       sep=",", 
                       header=None,
                       index_col=0,
                       dtype=float)
    # .T
    # data.columns = data.loc[0]
    # data= data[1:]
    return data


def read_data_test():
    data = read_data("C:\\Users\\e2204673\\Downloads\\timeseries\\50words\\50words_TRAIN")
    print(data)
    
def distance_euclidienne(point1 : list[float] , point2 : list[float] ) :
    if len(point1) != len(point2):
        raise ValueError("Points must be in the same dimensions")
    sum= 0
    for i in range(len(point1)):
        sum += (point1[i] - point2[i])**2
    return math.sqrt(sum)

def np_distance_test():
    data = read_data("C:\\Users\\e2204673\\Downloads\\timeseries\\50words\\50words_TRAIN")
    dist = np.linalg.norm(data.iloc[1] - data.iloc[2])
    print(dist)
    
def distance_euclidienne_test():
    data = read_data("C:\\Users\\e2204673\\Downloads\\timeseries\\50words\\50words_TRAIN")
    dist = distance_euclidienne(data.iloc[1].values, data.iloc[2].values)
    print(dist)
    
# @param point_idx - indice du point qu'on doit savoir ces distances des autres points
# @param K - proche voisin Nombre de voisin plus proche
def get_k_nearest_neighbors(data: pd.DataFrame, point: pd.Series, k: int):
    # Create a Series to store distances
    distances = pd.Series(index=data.index, dtype=float)
    # Compute distances between the selected point and all others
    for i in range(len(data)):
        distances.iloc[i] = distance_euclidienne(point.values, data.iloc[i].values)
    # Sort distances and get indices of k smallest
    nearest_neighbors = distances.nsmallest(k)
    # print(distances[1].index)
    
    return nearest_neighbors

def k1n_distance_euclideinne(training_set= 450) :
    # read_data_test()
    # np_distance_test()
    # distance_euclidienne_test()
    data = read_data("C:\\Users\\e2204673\\Downloads\\timeseries\\50words\\50words_TRAIN")
    test_data = read_data("C:\\Users\\e2204673\\Downloads\\timeseries\\50words\\50words_TEST")
    
    # print(data.index)
    # print(test_data.index)
    res = []
    correct = 0
    for i in range(len(test_data.head(training_set))) :
        knn = get_k_nearest_neighbors(data, test_data.iloc[i], 1)
        res.append(knn)
        # print(test_data.index[i]," == ",  knn.index[0])
        if test_data.index[i] == knn.index[0] : 
            correct += 1
    print(str((correct/ training_set) * 100 )+ "%")

def main():
    k1n_distance_euclideinne()
        
    

if __name__ == '__main__':
    main()
