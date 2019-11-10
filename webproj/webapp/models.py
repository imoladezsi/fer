import os

from django.db import models

from cnn.Constants import Constants

MODEL_CHOICES = [
    (os.path.join(Constants.BASE_PATH, os.path.join('example_model','DeXpression_CKL_BS128_LR1e-05_EP20')), 'DeXpression CK'),
]


class MainModel(models.Model):
    image = models.ImageField()
    predictor = models.CharField(max_length=20, choices=MODEL_CHOICES,
                                 #default='../svm/data_estimators/scikit_linearSVM_TP20_EP10_small_dataset.pkl'
                                )
