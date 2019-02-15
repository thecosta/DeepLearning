from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization

def BaselineModel():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=3072))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model
