class Utils(object):
    """
    Helper method to get the maximum prediction value
    """
    @staticmethod
    def get_max(prediction_result):
        emotions = prediction_result['emotions']
        numbers = prediction_result['result']
        label = emotions[0]
        max = numbers[0]
        for i in range(len(numbers)):
            if numbers[i] > max:
                max = numbers[i]
                label = emotions[i]
        return label

