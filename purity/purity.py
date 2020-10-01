# purity implementation for Clustering Class at Federal University of Uberlândia
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

def purity(y_pred,y_true):
  '''Purity is an external clustering evaluation algorithm and can be thought as
  an extent to which cluster contain a single class.

    #Arguments
      y_pred (list or np.array): the array-like structure containing the cluster
        information, usually is produced by a clustering algorithm;
      y_true (list or np.array): the array-like structure containing the label
        information about the data, usually known as y_train;
    #Returns
      results (float): the purity value of the data.
      summary (pd.DataFrame): the confusion matrix of the data.
    OBS.:
      results: bad cluster have purity near to 0, and perfect cluster have purity 1.
      summary: a good use is to print(summary) to get insight about the data information.
      overall: purity does not penalize for a high number of clusters.

  '''
  #transform the array like structures into a pandas dataframe, no need for assert because 
  #pandas dataframe expects an array like structure
  data = pd.DataFrame(y_pred, columns=["cluster(i)"])
  data["label(j)"] = y_true

  #variable initialization
  results = 0                       # summation variable
  M = len(data)                     # the total number of points
  K = data["cluster(i)"].nunique()  # the number of different clusters

  for i in range(K):
    # data["label(j)"] selects the label column of the dataframe
    # then we group the data by the cluster it belongs (data["cluster(i)"] == i)
    # then we count how many time each label appeared in that cluster (.value_counts())
    # then we get the maximum value, in other words, the number of the class which
    # appeared the most in that cluster (.max())
    results += data["label(j)"][data["cluster(i)"] == i].value_counts().max()

  # then we return the average correct labels found by the clusters,
  # and a summary for visual/manual confirmation
  return results / M, data.value_counts()



if __name__ == "__main__":

    
    if len(sys.argv) == 1:
        print("Invalid option, try python3 purity.py --help for instructions")
    
    elif sys.argv[1] == '--help':
        print("options:")
        print("--help : to show this options")
        print("--example : to run iris example")
    
    elif str(sys.argv[1]) == '--example':
        (X, y_true) = datasets.load_iris(True) #load iris data
        with open("results.txt", "w") as file_handler:
          file_handler.write("Purity execution for K Means, with k=[2,3,4]\n")
          for K in range(2,5): #execute for k=2,k=3 and k=4
            centroids, y_pred = k_means(K,X)
            result, summary = purity(y_pred,y_true)
            file_handler.write("* K = " + str(K) + "\n")
            file_handler.write("Purity: " + str(result) + "\n")
            file_handler.write("Summary: \n")
            file_handler.write(str(summary) + "\n\n")
        
        print('Done, saved in the directory!')

    else:
      print('Missing Arguments or incorrect arguments,')
      print('type: python3 purity.py --help for more options')
    
        