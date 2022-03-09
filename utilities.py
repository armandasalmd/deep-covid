import datetime
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import constants as CONSTANTS
from enums import ClassificationType, LabelType

def detect_GPU():
  if (len(tf.config.list_physical_devices('GPU'))>0):
    print("✅ GPU detected")
  else:
    print("❌ GPU was not found")

def merge_dataframe_dictionaries(dict1, dict2):
  keys_union = list(set(dict1.keys()) | set(dict2.keys()))
  result_dict = {}

  for key in keys_union:
    dfs = []
    
    if dict1[key] is not None:
      dfs.append(dict1[key])

    if dict2[key] is not None:
      dfs.append(dict2[key])

    result_dict[key] = pd.concat(dfs).reset_index()
  
  return result_dict

def flatten_dataframes_dictionary(dfDictionary):
  return pd.concat(dfDictionary.values()).reset_index()

def get_tensorboard_callback(model_type):
  log_dir = "logs/" +\
    model_type.value + "-" +\
    CONSTANTS.CLASSIFICATION_MODE.value + "/" +\
    datetime.datetime.now().strftime("%m%d-%H%M")

  return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,)

def min_count_balance_data_dict(dataframes_dict):
  min = 999999

  for classDf in dataframes_dict.values():
    if classDf.shape[0] < min:
      min = classDf.shape[0]

  for labelType, classDf in dataframes_dict.items():
    multiplier = 1

    if CONSTANTS.CLASSIFICATION_MODE == ClassificationType.TWO_CLASS and labelType == LabelType.NORMAL:
      multiplier = 2
    
    dataframes_dict[labelType] = classDf.sample(min * multiplier, random_state=42)

def load_my_model(model_type, classification_type = CONSTANTS.CLASSIFICATION_MODE):
  model_base_dir = "models/" +\
    model_type.value + "-" +\
    classification_type.value
  
  latest_model_folder = os.listdir(model_base_dir)[-1]
  path = model_base_dir + "/" + latest_model_folder
  print("Loading model from:", path)

  return load_model(path)

def save_my_model(model, model_type, plot_architecture):
  # /models/<modelType>-<classificationType>/<MM:DD-HH:mm>
  if model is None:
    return

  save_dir = "models/" +\
    model_type.value + "-" +\
    CONSTANTS.CLASSIFICATION_MODE.value + "/" +\
    datetime.datetime.now().strftime("%m%d-%H%M")
  
  if plot_architecture:
    f_location = save_dir + "/model_architecture.png"
    plot_model(model, show_shapes=True, to_file=f_location)
    print("Image saved to:", f_location)

  model.save(save_dir)

def categorical_covid_sensitivity(confusion_matrix):
  true_covid_row = confusion_matrix[0]
  return true_covid_row[0] / np.sum(true_covid_row)

def categorical_covid_specificity(confusion_matrix):
  true_normal_row = confusion_matrix[1]
  true_pneu_row = confusion_matrix[2]
  fp = true_normal_row[0] + true_pneu_row[0] # false positives count
  non_covid_sum = np.sum(true_normal_row) + np.sum(true_pneu_row)
  return (non_covid_sum - fp) / non_covid_sum
