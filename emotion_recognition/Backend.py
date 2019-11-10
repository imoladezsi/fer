import keras.backend as k

from cnn.ModelManagerCNN import ModelManagerCNN
from emotion_recognition.enums import EstimatorTypes, Methods
import os


class Backend(object):
    """
    Class that works as a wrapper around the two Managers and adapts their interfaces
    """
    def __init__(self, model_path):
        if EstimatorTypes.DeXpression.name in model_path:
            self.__manager = ModelManagerCNN(model_path, 50, 1)
        elif EstimatorTypes.SmallVGGNet.name in model_path:
            self.__manager = ModelManagerCNN(model_path, 64, 3)
        else:
            raise Exception("Unsupported type for model")

    def predict(self, image_path):
        """
        return the prediction to the image
        :param image_path: the path of the image to be estimated
        :return: a dictionary: {'emotions': the labels, 'result': the predictions, 'max': the winning class label}
        """
        k.clear_session()
        return self.__manager.estimate_emotion(image_path)

    def train(self):
        """
        performs the training and saves the results
        :return: None
        """
        self.__manager.train()


