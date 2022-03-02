DATASET_CHEST_ROOT = "./datasets/dataset1"
DATASET_RADIOGRAPHY_ROOT = "./datasets/dataset3"

TRAIN_IMAGE_GEN_OPTIONS = dict(
  rescale=1./255,
  rotation_range=20,
  width_shift_range=0.2,
  height_shift_range=0.2,
  horizontal_flip=True
)

TEST_IMAGE_GEN_OPTIONS = dict(rescale=1./255)

TEST_SPLIT_SIZE = 0.2