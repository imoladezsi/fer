from keras.callbacks import ReduceLROnPlateau
from keras.engine.saving import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras import activations

import matplotlib.pyplot as plt
import matplotlib
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from cnn.Constants import *
from emotion_recognition.EmotionRecognitionAPI import EmotionRecognitionAPI

# TODO: don't use reflection, it might be slow
from cnn.models.DeXpression import DeXpression
from cnn.models.SmallVGGNet import SmallVGGNet
from cnn.models.MnistCNN import MnistCNN
from cnn.models.MobileNet import MobileNet

matplotlib.use("Agg")


class ModelManagerCNN(EmotionRecognitionAPI):
    def __init__(self, model_path, dim, channel):
        super().__init__(model_path)  # load_model is called
        self.__dim = dim
        self.__channel = channel
        self.__model_path = model_path

    def __load_label_binarizer(self):
        print("[INFO] loading label binarizer...")
        lst = self._model_path.split(os.path.sep)
        lb = os.path.join(os.path.sep.join(lst[0:-1]), "label_binarizer_" + str(lst[-1]))
        mlb = pickle.loads(open(lb, "rb").read())
        return mlb

    def __train_core(self, model, train_x=None, train_y=None, test_x=None, test_y=None):

        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=Constants.INIT_LR)

        if train_x is not None and train_y is not None and test_x is not None and test_y is not None:
            try:
                return model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=Constants.EPOCHS,
                                 batch_size=Constants.BATCH_SIZE, callbacks=[reduceLR])
            except TypeError:
                return model.fit(train_x, train_y, n_epoch=Constants.EPOCHS, validation_set=Constants.TEST_PERCENTAGE,
                                 show_metric=True, batch_size=Constants.BATCH_SIZE, callbacks=[reduceLR])

        print("[INFO] Using data generator to fit")
        train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, )

        train_generator = train_datagen.flow_from_directory(
            Constants.DATASET_NAME,
            target_size=(Constants.IMG_DIM, Constants.IMG_DIM),
            batch_size=Constants.BATCH_SIZE,
            subset='training')  # set as training data

        validation_generator = train_datagen.flow_from_directory(
            Constants.DATASET_NAME,  # same directory as training data
            target_size=(Constants.IMG_DIM, Constants.IMG_DIM),
            batch_size=Constants.BATCH_SIZE,
            subset='validation')  # set as validation data

        model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples,  # batch_size
            validation_steps=validation_generator.samples,
            epochs=Constants.EPOCHS,
            validation_data=validation_generator)

    def __resume(self, model_name, train_x=None, train_y=None, test_x=None, test_y=None):
        model = load_model(model_name)
        return model, self.__train_core(train_x, train_y, test_x, test_y)

    def get_data(self):
        data = []
        labels = []
        image_paths = sorted(list(paths.list_images(Constants.DATASET_NAME)))
        if not image_paths:
            raise Exception("Empty dataset! ")
        random.seed(Constants.RANDOM_STATE)
        random.shuffle(image_paths)

        # loop over the input images
        for image_path in image_paths:
            # load the image, resize it to to a fixed value and store it
            image = cv2.imread(image_path)
            image = cv2.resize(image, (Constants.IMG_DIM, Constants.IMG_DIM))

            if Constants.CHANNEL == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image[..., np.newaxis]
            data.append(image)

            # extract the class label from the image path
            label = image_path.split(os.path.sep)[-2]
            labels.append(label)
        return data, labels

    def __transform_labels(self, train_y, test_y):
        lb = LabelBinarizer()
        train_y = lb.fit_transform(train_y)
        test_y = lb.transform(test_y)
        return lb, train_y, test_y

    def __save_figure(self, H):
        N = np.arange(0, Constants.EPOCHS)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.plot(N, H.history["acc"], label="train_acc")
        plt.plot(N, H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy " + Constants.SAVE_MODEL_AS)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(os.path.join(Constants.OUTPUT_PATH, Constants.SAVE_MODEL_AS + "_plot.png"))

    def __save_model_and_label_binarizer(self, model, lb, name):
        model.save_weights(os.path.join(Constants.OUTPUT_PATH, name + '.h5'))
        model.save(os.path.join(Constants.OUTPUT_PATH, name))

        print("[INFO] serializing network and label binarizer...")
        f = open(os.path.join(Constants.OUTPUT_PATH,"label_binarizer_" + name), "wb")
        f.write(pickle.dumps(lb))
        f.close()

    def train(self):

        data, labels = self.get_data()
        model = globals()[Constants.MODEL_NAME]().create_model(Constants.IMG_DIM, Constants.IMG_DIM, Constants.CHANNEL)

        # scale the raw pixel intensities to the range [0, 1]
        try:
            data = np.array(data, dtype="float") / 255.0
            labels = np.array(labels)
            # split the data into training and testing splits
            (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=Constants.TEST_PERCENTAGE,
                                                                  random_state=Constants.RANDOM_STATE)

            # convert the labels from integers to vectors (required for multiclass classification)
            # print("[INFO] class labels:")
            lb, train_y, test_y = self.__transform_labels(train_y, test_y)

            if Constants.TRAIN:
                H = self.__train_core(model, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
            else:
                model, H = self.__resume(os.path.join(os.path.join(Constants.BASE_PATH,"data_final_candidates" ), Constants.RESUME), train_x=train_x, train_y=train_y,
                                  test_x=test_x, test_y=test_y)
            self.__save_model_and_label_binarizer(model, lb, Constants.SAVE_MODEL_AS)
            self.__save_figure(H)

        except MemoryError:
            self.__train_core(model)

    def load_model(self):
        print("[INFO] loading network..")
        model_path = os.path.join(self._model_path)
        return load_model(model_path)

    def estimate_emotion(self, image_path):
        print("[INFO] estimating...")
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.__dim, self.__dim))

        if self.__channel == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]
        image = np.expand_dims(image, axis=0)
        # scale the pixel values to [0, 1]
        image = image.astype("float") / 255.0

        # load the label binarizer
        mlb = self.__load_label_binarizer()

        model = self.load_model()

        # make a prediction on the image
        preds = model.predict(image)

        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = mlb.classes_[i]

        return {'emotions': mlb.classes_, 'result': preds[0], 'max': label}

