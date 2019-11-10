import os

from cnn.Constants import Constants
from cnn.ModelManagerCNN import ModelManagerCNN


if __name__ == "__main__":
    mm = ModelManagerCNN(Constants.MODEL_PATH, Constants.IMG_DIM, Constants.CHANNEL)
    mm.train()
