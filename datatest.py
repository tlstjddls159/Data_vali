import os
import pandas as pd
import math
import numpy as np
import keras
from keras.layers import Bidirectional, LSTM, CuDNNLSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate, Flatten, CuDNNGRU
from keras.models import Sequential, Model
from datasetList import *

path = './mung_dong/'

x_train, x_val, y_train, y_val  = dataset(path)

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(1, 81))
# Embed each integer in a 128-dimensional vector
lstm1 = CuDNNLSTM(128, return_sequences=True)(inputs)
lstm2 = CuDNNLSTM(128)(lstm1)
outputs = Dense(19, activation=keras.activations.softmax)(lstm2)

model = Model(inputs, outputs)
model.summary()

learning_rate = 0.00001

model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_val, y_val, batch_size=4, epochs=1000, shuffle=True, validation_data=[x_train, y_train])