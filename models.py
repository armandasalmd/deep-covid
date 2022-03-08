from keras.layers import BatchNormalization, Dense, Flatten, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf

from enums import ModelType

def get(modelType):
  if modelType == ModelType.DEEP_COVID:
    return _getDeepCovidModel()
  if modelType == ModelType.MOBILE_NET_V2:
    return _getMobileNetV2()
  else:
    return None

def _getDeepCovidModel():
  model = Sequential([
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

  # move selected metrics to constants
  opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
  model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
  )
  return model


def _getMobileNetV2():
  base_model = MobileNetV2(input_shape = (256, 256, 3), include_top = False, weights = "imagenet")
  base_model.trainable = False

  model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.15),
    Dense(3, activation="softmax")                                     
  ])
  opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
  model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
  )

  return model

