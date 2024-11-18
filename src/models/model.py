from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class CarPriceModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None
    
    def fit(self, X, y):
        # Implementação do treinamento virá aqui
        pass
    
    def predict(self, X):
        # Implementação da previsão virá aqui
        pass
    
    def preprocess_data(self, data):
        # Implementação do pré-processamento virá aqui
        pass 