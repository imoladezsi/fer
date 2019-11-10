from enum import Enum


class Classes(Enum):
        ANGER = "anger"
        DISGUST = "disgust"
        FEAR = "fear"
        HAPPINESS = "happiness"
        NEUTRAL = "neutral"
        SADNESS = "sadness"
        SURPRISE = "surprise"


class EstimatorTypes(Enum):
        DeXpression = "DeXpression"
        SmallVGGNet = "SmallVGGNet"
        SVM = "SVM"


class Methods(Enum):
        CNN = "CNN"