from keras.engine.saving import load_model
from cnn.Constants import Constants
import numpy as np
import pickle
import cv2
import os
"""
script used to evaluate some or all the models from inside a directory on a dataset 
"""

test_directory = "data_output"
test_dataset = "dataset"
save_results = "data_evaluations"
filter = ""


def write(filename, content):
    with open(filename, 'w') as g:
        g.write(content)


def load_images():
    x_test = []
    y_test = []

    # loop over the input images
    for path, subdirs, files in os.walk(os.path.join(Constants.BASE_PATH,test_dataset)):
        for file in files:
            # load the image, resize it to 64x64 pixels and store the image in the
            # data list
            image_path = os.path.join(path, file)
            # load the image, resize it to to a fixed value and store it
            image = cv2.imread(image_path)
            image = cv2.resize(image, (Constants.IMG_DIM, Constants.IMG_DIM))

            if Constants.CHANNEL == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image[..., np.newaxis]
            x_test.append(image)
            # extract the class label from the image path and update the
            # labels list
            label = image_path.split(os.path.sep)[-2]
            y_test.append(label)
    x_test = np.array(x_test, dtype="float") / 255.0
    y_test = np.array(y_test)
    return x_test, y_test


def evaluate(x_test,y_test):
    try:
        lb = pickle.loads(open(os.path.join(Constants.BASE_PATH,test_directory + "/label_binarizer_" + model_name), "rb").read())
        y = lb.fit_transform(y_test)
        score = model.evaluate(x_test, y, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        write(os.path.join(Constants.BASE_PATH,save_results + "/" + model_name + "_" + test_dataset)
              , "Test loss: " + str(score[0]) + "\nTest accuracy: " + str(score[1]))
    except Exception as e:
        print("There was a problem evaluating the model")
        print(e)


if __name__ == "__main__":
    for f in os.listdir(os.path.join(Constants.BASE_PATH,test_directory)):
        if not (f.endswith(tuple(['jpg', 'png', 'h5'])) or f.startswith('label_binarizer')) and filter in f:
            print(f)
            model_name = f

            model = load_model(os.path.join(Constants.BASE_PATH, os.path.join(test_directory,f)))
            print("[INFO] model loaded")

            x_test, y_test = load_images()
            evaluate(x_test, y_test)



