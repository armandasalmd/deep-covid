import datetime
import os

from keras.layers import BatchNormalization, Dense, Flatten, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import load_model

from enums import ClassificationType, ModelType
import constants as CONSTANTS

def get(modelType):
  if modelType == ModelType.DEEP_COVID:
    return _getDeepCovidModel()
  if modelType == ModelType.MOBILE_NET_V2:
    return _getMobileNetV2()
  else:
    return None

def _getDeepCovidModel():
  return Sequential([
    BatchNormalization(),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(128, 3, activation="relu"),
    MaxPooling2D(),
    Conv2D(128, 3, activation="relu"),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(256, 3, activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.15),
    Dense(3, activation="softmax")
  ])

def _getMobileNetV2():
  base_model = MobileNetV2(input_shape = (256, 256, 3), include_top = False, weights = "imagenet")
  base_model.trainable = False

  return Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.15),
    Dense(3, activation="softmax")                                     
  ])

def save_my_model(model, model_type):
  # /models/<modelType>-<classificationType>/<MM:DD-HH:mm>
  if model is None:
    return

  save_dir = "models/" +\
    model_type.value + "-" +\
    CONSTANTS.CLASSIFICATION_MODE.value + "/" +\
    datetime.datetime.now().strftime("%m%d-%H%M")

  model.save(save_dir)

def load_my_model(model_type, classification_type = CONSTANTS.CLASSIFICATION_MODE):
  model_base_dir = "models/" +\
    model_type.value + "-" +\
    classification_type.value
  
  latest_model_folder = os.listdir(model_base_dir)[-1]

  return load_model(model_base_dir + "/" + latest_model_folder)
