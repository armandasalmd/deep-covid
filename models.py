from keras.layers import BatchNormalization, Dense, Flatten, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.densenet import DenseNet201
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

def precision(y_true, y_pred):
  """Precision metric.
  Only computes a batch-wise average of precision.
  Computes the precision, a metric for multi-label classification of
  how many selected items are relevant.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def _get_compile_options():
  IS_2C = CONSTANTS.CLASSIFICATION_MODE == ClassificationType.TWO_CLASS
  options = {
    "optimizer": tf.keras.optimizers.Adam(learning_rate = 0.001),
    "loss": "binary_crossentropy" if IS_2C else "sparse_categorical_crossentropy",
    "metrics": 
      [sensitivity, specificity, precision, "accuracy"] 
      if IS_2C 
      else ["accuracy"]
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
  elif modelType == ModelType.EFFICIENT_NET_REFINED:
    return _getEfficientNetB7_refined()
  elif modelType == ModelType.DENSE_NET_201:
    return _getDenseNet201()
  elif modelType == ModelType.MOBILE_NET_V2_REFINED:
    return _getMobileNetV2_refined()
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
    _get_output_layer() # Sigmoid or Softmax
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

def _getDenseNet201():
  base_model = DenseNet201(input_shape = tuple([*CONSTANTS.INPUT_SIZE, 3]), include_top = False, weights = "imagenet")
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

def _getMobileNetV2_refined():
  base_model = MobileNetV2(input_shape = tuple([*CONSTANTS.INPUT_SIZE, 3]), include_top = False, weights = "imagenet")
  base_model.trainable = False
  base_model.layers[-1].trainable = True

  model = Sequential([
    base_model,
    Dropout(0.15),
    Conv2D(256, 3, activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    _get_output_layer()
  ])

  model.compile(**_get_compile_options())
  return model

def _getEfficientNetB7_refined():
  base_model = EfficientNetB7(input_shape = tuple([*CONSTANTS.INPUT_SIZE, 3]), include_top = False, weights = "imagenet")
  base_model.trainable = False
  base_model.layers[-1].trainable = True

  model = Sequential([
    base_model,
    Dropout(0.15),
    Conv2D(128, 3, activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    _get_output_layer()
  ])

  model.compile(**_get_compile_options())
  return model
