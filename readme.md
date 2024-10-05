# iccnn

simple convolution neural network for image classification

## getting started

1. have `python >=3.11.0` installed in your respective platform
2. clone the repository
3. create a python virtual environment `python -m venv .venv`
4. activate the python virtual environment `source .venv/bin/activate`
5. make sure the python in your path is from the virtual environment
6. install pip dependencies `pip install -r requirements.txt`
7. run `python main.py`

### plan outline

#### Week 1: Environment Setup and Dataset Preparation

Person 1: Environment Setup

- Install necessary libraries (TensorFlow, Keras, PyTorch, etc.)
- Set up a shared repository (e.g., GitHub) for collaboration
- Ensure all team members have the correct environment

Person 2: Dataset Selection

- Search for appropriate datasets from the UCI Machine Learning Repository (ensure it’s not CIFAR-10)
- Download the dataset and prepare the data pipeline (e.g., loading, preprocessing, normalization)
- Split the dataset into training (60%), validation (20%), and test (20%) sets

Person 3: CNN Architecture Definition

- Define the CNN architecture with the layers mentioned (convolutional, max-pooling, fully connected, dropout, softmax)
- Decide the parameters (filter size, kernel size, activation functions, dropout rate) and justify them
- Share initial architecture design with the team for feedback

Person 4: Literature Review

- Research activation functions (ReLU, softmax) and justify their use
- Research optimizers (Adam vs. SGD) and loss functions (sparse categorical crossentropy) for justification
- Summarize findings for the team

Deliverables by End of Week 1:

- Environment set up and shared repo
- Get Everyone upto speed with git and collaboration
- Dataset selected, preprocessed, and split
- Initial CNN architecture defined
- Justification for activations, optimizers, and loss function prepared
