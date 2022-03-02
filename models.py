from keras.layers import BatchNormalization, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential

from enums import ModelType

def get(modelType):
  if modelType == ModelType.DEEP_COVID:
    return _getDeepCovidModel()
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
    Conv2D(256, 3, activation="relu"),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(512, 3, activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.15),
    Dense(3, activation="softmax")
  ])
