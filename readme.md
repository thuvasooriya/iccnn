# iccnn

simple convolution neural network for image classification

## getting started

1. have `python >=3.11.0` installed in your respective platform
2. clone the repository
3. create a python virtual environment `python -m venv .venv`
4. activate the python virtual environment `source .venv/bin/activate`
    for windows `.venv/bin/activate`
5. make sure the python in your path is from the virtual environment
6. install pip dependencies `pip install -r requirements.txt`
7. run `python main.py`

### plan outline

#### week 1: environment setup and dataset preparation

[thuva](https://github.com/thuvasooriya): environment setup

- install necessary libraries (tensorflow, keras, pytorch, etc.)
- set up a shared repository (e.g., github) for collaboration
- ensure all team members have the correct environment

person 2: dataset selection

- search for appropriate datasets from the uci machine learning repository (ensure it’s not cifar-10)
- download the dataset and prepare the data pipeline (e.g., loading, preprocessing, normalization)
- split the dataset into training (60%), validation (20%), and test (20%) sets

abithan: cnn architecture definition

- define the cnn architecture with the layers mentioned (convolutional, max-pooling, fully connected, dropout, softmax)
- decide the parameters (filter size, kernel size, activation functions, dropout rate) and justify them
- share initial architecture design with the team for feedback

person 4: literature review

- research activation functions (relu, softmax) and justify their use
- research optimizers (adam vs. sgd) and loss functions (sparse categorical crossentropy) for justification
- summarize findings for the team

deliverables by end of week 1:

- [x] environment set up and shared repo
- [ ] get everyone upto speed with git and collaboration
- [ ] dataset selected, preprocessed, and split
- [ ] initial cnn architecture defined
- [ ] justification for activations, optimizers, and loss function prepared

#### week 2: cnn model implementation and training

person 1: cnn model implementation

- implement the cnn architecture in code (using tensorflow/keras or pytorch)
- ensure the model uses the adam optimizer and sparse categorical crossentropy loss function

person 2: training and validation

- train the cnn model for 20 epochs with a learning rate of 0.001
- plot training and validation loss for each epoch
- evaluate the initial performance on validation data

person 3: hyperparameter tuning

- run experiments with different learning rates (0.0001, 0.001, 0.01, 0.1)
- plot training and validation loss for different learning rates
- select the best learning rate with justification

person 4: testing and initial evaluation

- evaluate the trained cnn model on the test set
- record test accuracy, confusion matrix, precision, and recall
- analyze the results

deliverables by end of week 2:

- [ ] cnn model implemented and trained
- [ ] training and validation loss plots for different learning rates
- [ ] test accuracy, confusion matrix, precision, and recall recorded
