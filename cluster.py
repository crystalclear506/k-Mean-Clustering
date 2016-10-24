#! /usr/local/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, getopt, os

def kmeanCluster(initial_centroids, points):
    if type(initial_centroids) == np.ndarray:
        initial_centroids = initial_centroids.tolist()
    
    # N-Dimensional Euclidean Distance
    def euclidean_dist(point_1, point_2):
        if (len(point_1) == len(point_2)):     
            dimensions = len(point_1)
        else:
            raise ValueError('Incompatible Input parameters')
        accumulated = 0
        for dim in range(dimensions):   
            accumulated = accumulated + ((point_1[dim] - point_2[dim])**2)
        return np.sqrt(accumulated)
    
    def sum_euclidean(points, new_points):
        sum_value = 0.;

        if len(points) == len(new_points):
            for index, point in enumerate(points):
                sum_value = sum_value + euclidean_dist(points[index], new_points[index])

            return sum_value
        else:
            raise ValueError('Two lists have an unequal number of elements')

        return None
    
    #N-Dimensional Centroid calculation
    def calculateNewCentroid(points, cen_type='mean'):
        least_dist = 1e99
        new_centroid = None
        dimensions = len(points[0])
        center_point = []
        
#       Median Centroid
        if (cen_type=='median'):
            for dim in range(dimensions):
                bound_upper, bound_lower = max(points[:, dim]), min(points[:, dim])
                center_point.append((bound_upper + bound_lower)/2.)
#       Mean Centroid
        if (cen_type=='mean'):
            for dim in range(dimensions):
                center_point.append(np.mean(points[:, dim]))
            
        for point in points:
            if (euclidean_dist(point, center_point) < least_dist):
                least_dist = euclidean_dist(point, center_point)
                new_centroid = point

        return new_centroid
        
    final_clusters = [None] #Funny work-around using list making recursive step references to this local variable instead of copying it
    no_iterations = [0]

    def cluster(initial_centroids, points):
        clustered_indexes, clustered_records = [], []
        for point in points:
            least_distClusterIndex = [None, 1e99]        
            for centroid_index, centroid in enumerate(initial_centroids):
                if (euclidean_dist(centroid, point) < least_distClusterIndex[1]):
                    least_distClusterIndex[0] = centroid_index
                    least_distClusterIndex[1] = euclidean_dist(centroid, point)

            clustered_records.append(point)
            clustered_indexes.append(least_distClusterIndex[0])
        
        clusterized = np.append(np.array(clustered_indexes)[np.newaxis].T, np.array(clustered_records), axis=1)

        cluster_indexes = set(clusterized[:, 0])

        clusters = []
        for cluster_index in cluster_indexes:
            clusters.append(clusterized[clusterized[:, 0] == cluster_index][:, 1:])
        
        new_initial_centroids = []
        for c in clusters:
            new_initial_centroids.append(calculateNewCentroid(c))
        
        #Recursive Step
        if(sum_euclidean(initial_centroids, new_initial_centroids) == 0):
            final_clusters[0] = clusters
            print "number of iterations : ", no_iterations[0]
            return new_initial_centroids
        else: 
            no_iterations[0] += 1
            return cluster(new_initial_centroids, points)        
    
    return cluster(initial_centroids, points), final_clusters[0]

args = sys.argv

data_points = np.array(pd.DataFrame.from_csv(args[1], index_col=None, header=None))

stdin_centroids = raw_input("Enter indexes of initial centroids (space separated) : ")
centroid_indexes = stdin_centroids.split(" ")

input_centroids = []
for centroid_index in centroid_indexes:
    input_centroids.append(data_points[int(centroid_index)])

output = kmeanCluster(input_centroids, data_points)

result = output[1][:]


#File Operations

if not os.path.exists("clustering_output"):
    os.makedirs("clustering_output")

for cluster_index, cluster in enumerate(result):
    df_result = pd.DataFrame(cluster)
    df_result.to_csv("clustering_output/output_cluster_%s.csv" % cluster_index, index=False, header=None)


plot_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']

if (result[0].shape[1] == 2): #Check Two Dimensions
    for cluster_index, cluster in enumerate(result):
        plt.scatter(cluster[:, 0], cluster[:, 1], color=np.random.rand(3,1), marker=plot_markers[int(np.random.random()*len(plot_markers))])

    plt.savefig("clustering_output/output_cluster_fig.png")
else:
    try:
        os.remove("clustering_output/output_cluster_fig.png")
    except OSError:
        pass
