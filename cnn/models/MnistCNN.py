from keras import losses, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from cnn.models.ModelInterface import ModelInterface


class MnistCNN(ModelInterface):
    """
    Simplest model often used with the MNIST dataset
    """
    @staticmethod
    def create_model(width, height, depth):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(width, height, depth)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        return model
