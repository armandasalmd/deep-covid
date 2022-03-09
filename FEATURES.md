### Implemented features:
- 7/3/2022
  - save trained model with models/save_my_model() func
  - load trained model with models/load_my_model() func
  - average image pixel analysis (sample 1000-2000 images and reference Central limit Thoerem)
  - preview predict notebook cell
  - create custom accuracy calculation function
  - create utility function to draw Confusion matrix for selected model
  - [visualize feature maps for Convolutional layers](https://www.kaggle.com/arpitjain007/guide-to-visualize-filters-and-feature-maps-in-cnn)
  - Notebook cell with constants for "run options". Goal is to toogle options and "Run All" notebook cells to perform a task
  - experiment using lower precision variables (flaot16) to reduce memory usage. Save screenshots of evidence. How does it effect traning speed (compare before and after)
    - Investigation shows that no improvement nor loss is obtained
- 8/3/2022
  - Fix unable to plot confussion matrix for 2C labels
- 9/3/2022
  - Add model for transfer learning - `Efficient Net`
  - Add model for transfer learning - `Mobile Net V2`
  - Implement early stopping callback
    `es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)`
  - When plotting random 9 images, show 3 for each class instead and add label
  - In notebook plot trained model accuracy after training
  - add more metrics to model when training

### In progress:

### Features to deliver:
- Keras Tuner to tune my models
- [Model does not learn example](https://www.guru99.com/tensorboard-tutorial.html)
- Add model for transfer learning - `Alex Net`
  - Special installation required [here](https://github.com/heuritech/convnets-keras#get-the-weights-of-the-pre-trained-networks)
- [5. Ben Graham's Method](https://www.kaggle.com/sana306/detection-of-covid-positive-cases-using-dl) 
- [EDA mean vs std for positive/negative scatterplot](https://www.kaggle.com/sana306/detection-of-covid-positive-cases-using-dl)
- cleaning outlier images in dataset? Is that possible? Experiment. How does it improve accuracy?
- Generate logs for
  - Deep Covid
    - 2C initial model
    - 2C fine tuned model
    - 3C initial model
    - 3C fine tuned model
    - [Layer 1 - BatchNormali](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/)
    - [Batch normalization example](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/)
    - [ARVIX paper for batch normalisation](https://arxiv.org/abs/1502.03167)
  - Alex Net
    - 2C initial model
    - 2C fine tuned model
    - 3C initial model
    - 3C fine tuned model
  - Efficient Net
    - 2C initial model
    - 2C fine tuned model
    - 3C initial model
    - 3C fine tuned model
  - Mobile Net V2
    - 2C initial model
    - 2C fine tuned model
    - 3C initial model
    - 3C fine tuned model



> Note that the best way to monitor your metrics during training is via TensorBoard.
[Keras DOCS](https://keras.io/api/metrics/)