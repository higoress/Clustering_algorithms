import numpy as np
import pandas as pd
import time 

def replace_avg(data, column_name):
  avg = 0
  count = 0
  for data_value in data[column_name]:
    if (not pd.isna(data_value)):
      avg += data_value
      count += 1
  avg = avg // count
  new_column = []
  for data_value in data[column_name]:
    if pd.isna(data_value):
      new_column.append(avg)
    else :
      new_column.append(data_value)
  
  return new_column

def replace_mode(data, column_name):
  mode = data[column_name].mode()
  new_column = []
  for data_item in data[column_name]:
    if pd.isna(data_item):
      new_column.append(mode)
    else:
      new_column.append(data_item)
    
  return new_column

def input_missing_values(file_name, column_name, categorical=False):
  ''' data: string containing path + filename
      column_name: type string
      categorical: False for avg input, True for mode input
  '''
  data = pd.read_csv(file_name)


  if data[column_name].isna().sum() == 0:
    raise TypeError('This column does not have missing values')
  
  if categorical:
     data[column_name] = replace_mode(data,column_name)
  else:
    data[column_name] = replace_avg(data,column_name)


  data.to_csv('new_data.csv')


def input_missing_values2(file_name):
  ''' filename: string containing path + filename '''
  data = pd.read_csv(file_name)
  
  for column_name in data.columns:
    
    if data[column_name].isna().sum() == 0:
      print(column_name, ': doesnt have missing values')
    else:
      if data[column_name].dtype == 'int64' or data[column_name].dtype == 'float64':
        data[column_name] = replace_avg(data,column_name)
      else:
        data[column_name] = replace_mode(data,column_name)
  
  data.to_csv('result.csv')

data_path = 'data/fatal-police-shootings-data.csv'

input_missing_values(data_path, 'age')

input_missing_values(data_path, 'armed', categorical=True)

input_missing_values2(data_path)