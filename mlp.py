import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

class MultiLayerPerceptron:
    def __init__(self, x_train, y_train):
        self.x_train = x_train    
        self.y_train = y_train
        self.input_dim = self.x_train.shape[1]    
        self.model = self._define_model()

    
    def _define_model(self):
        N_NEURONS1 = 128
        DROPOUT_RATE1 = 0.15
        N_NEURONS2 = 128
        DROPOUT_RATE2 = 0.15
        N_CLASS = 10
        model = Sequential()
        model.add(Dense(N_NEURONS1, input_dim=self.input_dim))
        model.add(Activation("relu"))
        model.add(Dropout(DROPOUT_RATE1))
        model.add(Dense(N_NEURONS2))
        model.add(Activation("relu"))
        model.add(Dropout(DROPOUT_RATE2))
        model.add(Dense(N_CLASS))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model

    def fit(self, epochs=10, batch_size=16):
        self.model.fit(
            self.x_train, 
            np_utils.to_categorical(self.y_train), 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.1, 
            verbose=2
            )

    def predict(self, x_test):
        values = self.model.predict_classes(x_test, verbose=0)
        return values

