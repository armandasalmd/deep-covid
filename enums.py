from enum import Enum

class CollectionType(Enum):
  TEST = "test"
  TRAIN = "train"

class ClassificationType(Enum):
  TWO_CLASS = "2C"
  THREE_CLASS = "3C"

class ModelType(Enum):
  DEEP_COVID = "deepCovid"
  DENSE_NET_201 = "denseNet201"
  EFFICIENT_NET = "efficientNet"
  EFFICIENT_NET_REFINED = "efficientNetRefined"
  MOBILE_NET_V2 = "mobileNetV2"
  MOBILE_NET_V2_REFINED = "mobileNetV2Refined"

class DatasetType(Enum):
  COMBINED = "combined"
  CHEST = "chest"
  RADIOGRAPHY = "radiography"

class LabelType(Enum):
  COVID = "covid"
  NORMAL = "normal"
  PNEUMONIA = "pneumonia"