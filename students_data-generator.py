#! /usr/local/bin/python

import scipy.stats as stats
import sys
import numpy as np
import pandas as pd

def dataGen(n_points, n_synthetic_clusters=2, std=0.035):
    def truncated_normal(lower, upper, mean, std):
        return stats.truncnorm.rvs((lower-mean)/std,(upper-mean)/std,loc=mean,scale=std)
        
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
                    truncated_normal(0, 100, origin[0], std*100),
                    truncated_normal(200, 800, origin[1], std*600),
                    1 if (np.random.rand() > 0.5) else 0,
                    truncated_normal(500, 10000, origin[3], std*9500)
                ]
                return row
            
            arr.append(randRow())
            
    return np.array(arr)

args = sys.argv

print "points : ", args[1]
print "clusters :", args[2]
print "std :", args[3]

df = pd.DataFrame(dataGen(int(args[1]), n_synthetic_clusters=int(args[2]), std=float(args[3])))
df.to_csv("student_data.csv", index=False, header=None)