## Deep Covid - machine learning models predicting Covid-19

> Deep Covid is a custom given name and refers to Convolutional Neural Network model. Some other transfer learning models have been used for classification task: Normal, Pneumonia, Covid-19

#### Models used for task evaluation
1. DeepCovid
1. MobileNetV2
1. DenseNet201
1. EfficientNetB7

## Running the project (guide)

### Prerequisites

1. [Install Python](https://www.python.org/downloads/release/python-388/) (originally `Python3.8.8 64-bit` was used and is recommended)
    - Verify installation with `python --version`
2. [Install VS Code IDE](https://code.visualstudio.com/download) and `Jupyter` extension (alternativelly [Anaconda](https://docs.anaconda.com/anaconda/install/index.html))
3. Install the following Python packages (`pip install <package-name>`):
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - keras
    - tensorflow-gpu
4. Clone this repository and download datasets 
    - In folder `datasets` create `dataset1` and `dataset3` empty subfolders
    - Unzip the contents from the below dataset sources and place it in corresponding folders
    - **Dataset1** Chest X-ray (Covid-19 & Pneumonia) [(SOURCE)](https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia)
    - ~~Dataset2~~ - was dropped out during research
    - **Dataset3** COVID-19 Radiography Database [(SOURCE)](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

> ❗ More detailed dataset information can be found in `datasets/DATASETS.md` ❗

### Project structure

Repetitive utility functions, constants, enum types, and models have been placed to individual files (Separation of Concern principle). This helps to make the main notebook `deepCovidIndex.ipynb` more intuitive and easier to read/work with.

In `models.py` you can change each model's architecture. Training process can be executed by selecting RUN OPTIONS in `deepCovidIndex.ipynb:CELL2` and clicking run all button. Let's describe RUN OPTIONS now:

```python
# RUN OPTIONS
# Skips EDA, plot, chart analysis on dataset
SKIP_ANALYSIS = True 
# Delete intermediate variables used during dataset loading
SKIP_MEMORY_CLEANUP = False 
# Run long running training? 
SKIP_MODEL_TRAINING = False 
# Or load trained model from memory?
SKIP_MODEL_LOADING = not SKIP_MODEL_TRAINING 
# Random 500 samples used to predict 
SKIP_CUSTOM_ACCURACY = True 
# Plot confusion matrix?
SKIP_CONFUSSION_MATRIX = False 

# Train epoch count
EPOCHS = 20 
# Stop training if model is not improving
ENABLE_EARLY_STOP = True 
# Output metrics in /logs for TensorBoard
ENABLE_TRAIN_LOGS = True 
# Save trained weights
SAVE_TRAINED_MODEL = True 
# Model used for evaluation
SELECTED_MODEL = ModelType.EFFICIENT_NET_REFINED 
```

> ❗ Additionally, in `constants.py` you can change 2-class/3-class accuracy option. **2-class accuracy** refers to joining Pneumonia and Covid labels to a single class and classifying both of them as COVID-19 positive. ❗

After all options are configured, press **Run all cells** 