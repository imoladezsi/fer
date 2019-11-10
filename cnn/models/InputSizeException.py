class InputSizeException(Exception):
    """
    Exception to be thrown when the input size for a model is fixed, but it is not respected
    """
    def __init__(self, width, height, channel):
        self.message = "The proper input size should be ({},{},{}) - (width, height, channel)"\
            .format(width, height, channel)

