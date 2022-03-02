from enum import Enum

class CollectionType(Enum):
  TEST = "test"
  TRAIN = "train"

class ClassificationType(Enum):
  TWO_CLASS = "twoClassClassification"
  THREE_CLASS = "threeClassClassification"

class ModelType(Enum):
  DEEP_COVID = "deepCovid"
  ALEX_NET = "alexNet"
  EFFICIENT_NET = "efficientNet"
  MOBILE_NET_V2 = "mobileNetV2"

class DatasetType(Enum):
  COMBINED = "combined"
  CHEST = "chest"
  RADIOGRAPHY = "radiography"

class LabelType(Enum):
  COVID = "covid"
  NORMAL = "normal"
  PNEUMONIA = "pneumonia"