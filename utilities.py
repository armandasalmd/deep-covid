import datetime
import pandas as pd
import tensorflow as tf

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

  return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

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