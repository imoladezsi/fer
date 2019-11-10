class EmotionRecognitionAPI(object):
    def __init__(self, model_path):
        self._model_path = model_path

    def estimate_emotion(self, image_path):
        raise NotImplementedError("You have not implemented this method")

    def train(self):
        raise NotImplementedError("You have not implemented this method")

    def get_data(self):
        raise NotImplementedError("You have not implemented this method")

    def load_model(self):
        raise NotImplementedError("You have not implemented this method")

    def evaluate(self):
        print("There is no evaluation method for this approach")

    def visualizations(self):
        print("There are no data_visualizations for this approach")
