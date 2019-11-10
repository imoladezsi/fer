import os

def get_path():
    abs_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(abs_path)
    return file_dir

class Constants:
    """
    class containing the hyperparameters useful parameters that were adjusted during the
    training of the CNN models and and utilities
    """
    DROPOUT = 0.9
    BATCH_SIZE = 128
    DATASET_NAME = "dataset"
    BASE_PATH = get_path()
    OUTPUT_PATH = os.path.join(get_path(), "data_output")
    DATASET_PATH = os.path.join(get_path(), DATASET_NAME)
    INIT_LR = 100
    EPOCHS = 50
    TEST_PERCENTAGE = 0.33
    RANDOM_STATE = 25
    IMG_DIM = 50
    CHANNEL = 1
    MODEL_NAME = "DeXpression"
    SAVE_MODEL_AS = MODEL_NAME + "_" + DATASET_NAME + "_BS" + str(BATCH_SIZE) + "_LR" + str(INIT_LR) + "_EP" + str(
        EPOCHS) + "_DP" + str(DROPOUT) + "TP" + str(TEST_PERCENTAGE)
    MODEL_PATH = os.path.join(OUTPUT_PATH, SAVE_MODEL_AS)
    TRAIN = True
    PREDICT_IMAGE = " image name "
    RESUME = "model name"

