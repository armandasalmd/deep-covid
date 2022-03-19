from keras.layers import BatchNormalization, Dense, Flatten, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.efficientnet import EfficientNetB7
from keras import backend as K
import tensorflow as tf

import constants as CONSTANTS
from enums import ClassificationType, ModelType

def sensitivity(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
  true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
  possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
  return true_negatives / (possible_negatives + K.epsilon())

def _get_compile_options():
  IS_2C = CONSTANTS.CLASSIFICATION_MODE == ClassificationType.TWO_CLASS
  options = {
    "optimizer": tf.keras.optimizers.Adam(learning_rate = 0.001),
    "loss": "binary_crossentropy" if IS_2C else "sparse_categorical_crossentropy",
    "metrics": [sensitivity, specificity, "accuracy"] if IS_2C else ["accuracy"]
  }
  return options

def _get_output_layer():
  if CONSTANTS.CLASSIFICATION_MODE == ClassificationType.TWO_CLASS:
    return Dense(1, activation="sigmoid")
  else:
    return Dense(3, activation="softmax")

# ============================================

def get(modelType):
  if modelType == ModelType.DEEP_COVID:
    return _getDeepCovidModel()
  elif modelType == ModelType.MOBILE_NET_V2:
    return _getMobileNetV2()
  elif modelType == ModelType.EFFICIENT_NET:
    return _getEfficientNetB7()
  elif modelType == ModelType.ALEX_NET:
    return _getAlexNet()
  else:
    return None

def _getDeepCovidModel():
  model = Sequential([
    BatchNormalization(),
    Conv2D(64, 3, activation="relu", padding="same"),
    MaxPooling2D(),
    Dropout(0.25),
    Conv2D(256, 3, activation="relu"),
    MaxPooling2D(),
    Conv2D(256, 3, 2, activation="relu"),
    MaxPooling2D(),
    Conv2D(128, 3, 2, activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(48, activation="relu"),
    Dropout(0.1),
    _get_output_layer()
  ])


  model.compile(**_get_compile_options())
  return model

def _getMobileNetV2():
  base_model = MobileNetV2(input_shape = tuple([*CONSTANTS.INPUT_SIZE, 3]), include_top = False, weights = "imagenet")
  base_model.trainable = False

  model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    _get_output_layer()
  ])

  model.compile(**_get_compile_options())
  return model

def _getEfficientNetB7():
  base_model = EfficientNetB7(input_shape = tuple([*CONSTANTS.INPUT_SIZE, 3]), include_top = False, weights = "imagenet")
  base_model.trainable = False

  model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    _get_output_layer()
  ])

  model.compile(**_get_compile_options())
  return model

def _getAlexNet():
  # https://www.mydatahack.com/building-alexnet-with-keras/
  # https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
  # https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
  # https://github.com/dandxy89/ImageModels
  model=Sequential([
    Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation="relu", input_shape=tuple([*CONSTANTS.INPUT_SIZE, 3])),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3,3)),
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(1024,activation="relu"),
    Dropout(0.5),
    Dense(1024,activation="relu"),
    Dropout(0.5),
    _get_output_layer()
  ])

  model.compile(**_get_compile_options())
  model.summary()
  return model