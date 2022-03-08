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

### In progress:

### Features to deliver:
- explore Tensorboard and check if we can log "Confussion matrixes"
  - [Model does not learn example](https://www.guru99.com/tensorboard-tutorial.html)
- add more metrics to model when training
- Add model for transfer learning - `Alex Net`
- Add model for transfer learning - `Efficient Net`
- Add model for transfer learning - `Mobile Net V2`
- Keras Tuner to tune my models
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
