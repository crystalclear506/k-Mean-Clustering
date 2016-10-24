#! /usr/local/bin/python

import scipy.stats as stats
import sys
import numpy as np
import pandas as pd

def dataGen(n_points, n_synthetic_clusters=2, spread=0.035):
    def truncated_normal(lower, upper, mean, spread):
        return stats.truncnorm.rvs((lower-mean)/spread,(upper-mean)/spread,loc=mean,scale=spread)
        
    synthetic_cluster_origins = []
    
    for i in range(n_synthetic_clusters):
        synthetic_cluster_origins.append([np.random.rand()*100, #Math Score (0 to 100)
                    200+np.random.rand()*(800-200), #SAT Score (200 to 800)
                    1 if (np.random.rand() > 0.5) else 0, #Gender (Discrete 1 or 0)
                    500+np.random.rand()*(10000-500)])
    
    arr = []
    for i in range(n_points):
        for origin in synthetic_cluster_origins:
            def randRow():
                row = [
                    truncated_normal(0, 100, origin[0], spread*100),
                    truncated_normal(200, 800, origin[1], spread*600),
                    1 if (np.random.rand() > 0.5) else 0,
                    truncated_normal(500, 10000, origin[3], spread*9500)
                ]
                return row
            
            arr.append(randRow())
            
    return np.array(arr)

args = sys.argv

if (len(args) == 4):
    print "points for each cluster: ", args[1]
    print "clusters :", args[2]
    print "spread percentage:", args[3]

    spread = float(args[3])/100.0
    df = pd.DataFrame(dataGen(int(args[1]), n_synthetic_clusters=int(args[2]), spread=float(spread)))
    df.to_csv("student_data.csv", index=False, header=None)

elif(len(args) == 1):
    points = raw_input("Points for each cluster : ")
    clusters = raw_input("Clusters : ")
    spread_percentage = raw_input("Spread percentage :")

    spread = float(spread_percentage)/100.0
    df = pd.DataFrame(dataGen(int(points), n_synthetic_clusters=int(clusters), spread=float(spread)))
    df.to_csv("student_data.csv", index=False, header=None)

