class ModelInterface(object):
    """
    There is no clear way to enforce this constraint
    This class stands as a reminder for the structure the models adhere to
    """
    @staticmethod
    def create_model(width, height, depth):
        raise NotImplementedError("Method must be implemented by all models")
