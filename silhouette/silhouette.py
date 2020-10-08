# simplified silhouette widtch coefficient implementation for Clustering Class at 
# Federal University of Uberlândia
# @authors: higor, domitila, gustavo, anderson

import numpy as np
import pandas as pd
import random
import sys
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')


''' ------------------------------------------- UTILS -------------------------------------------------'''
# distance function developt in the last exercise
def distances(X,Y):
  '''Calculate euclidean and manhattan distance
      of two arrays
      
      #Arguments
        X (np.ndarray or list): the first object to calculate distance of;
        Y (np.ndarray or list): the second object to calculate distance of.
      #Returns
        euclidean_distance (float): the euclidean distance or L2 norm of the objects;
        manhattan_distance (float): the manhattan distance or L1 norm of the objects.
  '''
  #transform to np.ndarray if is not
  x = np.array(X)
  y = np.array(Y)

  euclidean_distance = np.sqrt(np.sum(np.power(np.abs(x-y),2)))
  manhattan_distance = np.sum(np.abs(x-y))

  return euclidean_distance,manhattan_distance

''' ---------------------------------- END OF UTILS ------------------------------------------------ '''

def k_means(K, data, distance="euclidean", max_iterations=100, variational=False):
  '''K-means is and algorithm to create K cluster using mean distance between
      the data point and the center of the cluster.

      #Arguments
        K (int): how many clusters the algorithm should divide the data;
        data (ndarray): ndarray with the data to cluster;
        distance (string): "euclidean" or "manhattan" as distance measurement;
        max_iterations (int): the number of iterations the algorithm should run 
        before stopping.
        variational (bool): if algorithm should stop if the clusters don't change
      #Returns
        cluster_centroids (list): coordinate of the cluster centers
        labels (list): the cluster label of each point

      OBS.: 
            * the returning parameters was inspired by sklearn implementation of
              k_means, but the algorithm was implemented by the group members.
            * the data must not have missing values
            * due to the unsupervised behavior, must not have label information
  '''

  assert type(K) == int, 'The number of clusters, K, must be an integer'
  assert type(data) == np.ndarray, 'the data type must be a numpy.ndarray'
  assert distance in ['euclidean','manhattan'], 'the distance must be euclidean or manhattan'
  assert type(max_iterations) == int, 'The max_iterations value must be an integer'

  clusters = {}
  distance_option = 0 if distance == 'euclidean' else 1 
  random.seed()

  #initialize cluster data objects
  clusters['centroids'] = random.sample(data.tolist(),K)
  clusters['cluster'] = [[] for _ in range(K)]
  clusters['variations'] = [0 for _ in range(K)]
  clusters['labels'] = []

  for iteration in range(max_iterations):
    aux_distances = []
    variation = []
    clusters['labels'].clear()
    for elemnt in data:
      for centroid in clusters['centroids']: 
        aux_distances.append(distances(elemnt,centroid)[distance_option]) #calculate the distance to all centroids
      clusters['cluster'][np.argmin(aux_distances)].append(elemnt) #add the element to the nearest centroid
      clusters['labels'].append(np.argmin(aux_distances))    #add the label of the cluster to the array
      aux_distances.clear()
    for i in range(K):
      clusters['centroids'][i] = np.mean(clusters['cluster'][i], axis=0)  #update clusters centroids, axis=0 return mean element wise
      clusters['cluster'][i].clear()                                      #clear cluster data to new iteration
      variation.append(np.mean(clusters['cluster'][i]))                   #calculate cluster variation to watch stop criteria
      
    if variational:
      if variation == clusters['variations']: break                         #if cluster variation remain the same, break
      else: clusters['variations'] = variation                              #update variation data

  return clusters['centroids'],clusters['labels']

def silhouette_width_simplified(X, centroids, y_pred, distance="euclidean"):
  ''' Simplified Silhouette Width Coefficient(SSWC) is a measure of how similar 
  an object is to it's on clustering (cohesion), compared to other 
  clusters(separation).
  SSWC ranges in [-1,1] where 1 indicate the objects are a good match to their
  clusters and -1 otherwise.

  #Arguments
    X (array-like): array-like with the objects data;
    centroids (array-like): an array-like containing the centroids objects;
    y_pred (array-like): an array-like object containing information about the 
    cluster a given object;
      belongs;
    distance (string): "euclidean" or "manhattan" to select distance measurement
      option.
    #Returns
      coefficient (float): the SSWC value for the given data input; 
      SSWC belongs to [-1,1].
  '''
  assert distance in ['euclidean','manhattan'],'the distance must be euclidean or manhattan'
  assert len(centroids) > 1 and len(centroids) < len(X)-1,'centroids length must be in range of: 1 < len(centroids) < len(X)- 1 '
  
  #initialization
  N = len(X)      #sample size
  summation = 0   #summation
  distance_option = 0 if distance == 'euclidean' else 1 
  
  for i in range(N):
    #calculate the distance between the i_objetc to all other centroids
    avg_distances = [distances(X[i], centroid)[distance_option] for centroid in centroids]

    #set the intra centroid distance, note: pop remove the select object from the
    #list, and pop function removes by it's index;
    #example a = [2,4,3], a.pop(2) will return 3.
    a_i = avg_distances.pop(y_pred[i])

    #set the minimum inter centroid distance
    b_i = min(avg_distances)
    
    #calculate the coefficient and sum with the others values;
    summation += ((b_i - a_i)/max(a_i,b_i))

  #return the average of the summation as the coefficient.
  return (summation / N)


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print("Invalid option, try python3 silhouette.py --help for instructions")
    
    elif sys.argv[1] == '--help':
        print("options:")
        print("--help : to show this options")
        print("--example : to run iris example")
    
    elif str(sys.argv[1]) == '--example':
        (X, _) = datasets.load_iris(True) #load iris data
        with open("results.txt", "w") as file_handler:
          file_handler.write("SSWC execution for K Means, with k=[2,3,4]\n")
          for K in range(2,5): #execute for k=2,k=3 and k=4
            centroids, y_pred = k_means(K,X)
            coeff = silhouette_width_simplified(X, centroids, y_pred)
            file_handler.write("* K = " + str(K) + "\n")
            file_handler.write("SSWC: " + str(coeff) + "\n\n")
        
        print('Done, saved in the directory!')

    else:
      print('Missing Arguments or incorrect arguments,')
      print('type: python3 silhouette.py --help for more options')
    
        