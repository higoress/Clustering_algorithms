import numpy as np
import pandas as pd
import sys
#from sklearn import datasets

# letra a)
# (x,_) = datasets.load_iris(True) easy way to load iris dataset from sklearn
# but in this case, we're gonna use the data coming from the data/iris.csv

# letra b)
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

# letra c)
def distance_matrix(data,metric):
  '''Compute the distance matrix of the dataset (all against all)
      
      #Arguments
        data (numpy.ndarray): the rows and columns containing the data
        to compute the distances
        metric (string): 'euclidean' or 'manhattan', the type of the norm

      #Returns
        distance_matrix (numpy.ndarray): the distance matrix of the objective
  '''
  if metric == 'euclidean':
    kind = 0
  elif metric == 'manhattan':
    kind = 1
  else:
    raise ValueError("Unsupported kind of norm at metric, try 'eucliean' or 'manhattan'")
  matrix = []
  row_length = data.shape[0]
  for i in range(row_length):
    row = []
    for j in range(row_length):
      row.append(distances(data[i],data[j])[kind])
    matrix.append(row)
  
  return np.array(matrix)

# letra d
def save_on_file(matrix,file_name='results'):
  '''Save the distance matrix into csv file

      #Arguments
        matrix (np.ndarray): the data to be saved in the csv file
        file_name (string): the name of the resulted csv. default='results.csv'
      #Returns
        No return
  '''
  name = file_name + '.csv'

  np.savetxt(name, matrix, delimiter=',', fmt='%0.5f')




if __name__ == "__main__":

    if sys.argv[1] == '--help':
        print("options:")
        print("--help : to show this options")
        print("--example : to run iris example")
        print("file metric [name]: where ")
        print("\t file is the relative path of the file,")
        print("\t metric can be manhattan or euclidean,")
        print("\t [name] the resulting csv name (without .csv), default=results.csv")
    
    elif str(sys.argv[1]) == '--example':
        data = pd.read_csv('data/iris.csv', header=None) #the dataset has no header names
        data.drop(columns=data.columns[-1], inplace=True) # drop the name of the flowers from the dataframe
        matrix_of_distances = distance_matrix(data.values,'euclidean')
        save_on_file(matrix_of_distances,'results_iris_example')
        print('Done, saved in the directory!')

    else:
        if len(sys.argv) < 3:
            print('Missing Arguments or incorrect arguments,')
            print('type: python3 lab4.py --help for more options')
        elif len(sys.argv) == 3:
            file_path = sys.argv[1]
            metric = sys.argv[2]
            data = pd.read_csv(file_path, header=None)
            matrix_of_distances = distance_matrix(data.values, metric)
            save_on_file(matrix_of_distances)
        else:
            file_path = sys.argv[1]
            metric = sys.argv[2]
            result_name = sys.argv[3]
            data = pd.read_csv(file_path, header=None)
            matrix_of_distances = distance_matrix(data.values, metric)
            save_on_file(matrix_of_distances, result_name)



