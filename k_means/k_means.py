# k_means implementation for Clustering Class at Federal University of Uberlândia
# @authors: higor, domitila, gustavo, anderson

import numpy as np
import pandas as pd
import random
import sys
#from sklearn import datasets #only if using iris example


'''         UTILS: 
'''
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


def save_on_file(matrix,file_name='results'):
  '''Save the distance matrix into csv file

      #Arguments
        matrix (np.ndarray): the data to be saved in the csv file
        file_name (string): the name of the resulted csv. default='results.csv'
      #Returns
        No return
  '''
  name = file_name + '.csv'

  np.savetxt(name, matrix, delimiter=',', fmt='%0.1f')


# END OF UTILS

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
        cluster_centers (list): coordinate of the cluster centers
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

  #initialize cluster data objects
  clusters['centroids'] = random.choices(data,k=K)
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



if __name__ == "__main__":

    
    if len(sys.argv) == 1:
        print("Invalid option, try --help for instructions")
    
    elif sys.argv[1] == '--help':
        print("options:")
        print("--help : to show this options")
        print("--example : to run iris example")
        print("--interactive : to run the script interactively")
        print("sysarg : execute as")
        print("python3 k_means.py FILE_PATH K [NAME]: where ")
        print("\t FILE_PATH is the relative path of the file,")
        print("\t K is the number of the clusters,")
        print("\t [NAME] the resulting csv name (without .csv), default=results.csv")
        print("eg.: \t python3 k_means.py data/iris_unlabelled.csv 3")
        print("or:  \t python3 k_means.py data/iris_unlabelled.csv 3 iris_results")
    
    elif str(sys.argv[1]) == '--example':
        data = pd.read_csv('data/iris_unlabelled.csv', header=None) 
        centroids, labels = k_means(3,data.values)
        save_on_file(labels,'results_iris_example')
        print('Done, saved in the directory!')

    elif str(sys.argv[1]) == '--interactive':
        file_path = input('Insert the file path:')
        clusters = int(input('Insert the cluster number(K):'))
        data = pd.read_csv(file_path, header=None)
        centroids,labels = k_means(clusters,data.values)
        save_on_file(labels,'results_iris_interactive')
        print('Done, saved in the directory!')

    else:
        if len(sys.argv) < 3:
            print('Missing Arguments or incorrect arguments,')
            print('type: python3 lab4.py --help for more options')
        elif len(sys.argv) == 3:
            file_path = sys.argv[1]
            clusters = int(sys.argv[2])
            data = pd.read_csv(file_path, header=None)
            centroids,labels = k_means(clusters,data.values)
            save_on_file(labels)
            print('Done, saved in the directory!')
            
        else:
            file_path = sys.argv[1]
            clusters = int(sys.argv[2])
            result_name = sys.argv[3]
            data = pd.read_csv(file_path, header=None)
            centroids,labels = k_means(clusters,data.values)
            save_on_file(labels, result_name)
            print('Done, saved in the directory!')