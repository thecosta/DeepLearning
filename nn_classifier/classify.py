import keras
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from models.BaselineModel import BaselineModel


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    #Import data
    x_train = np.reshape(x_train, (x_train.shape[0], -1)) # 50000, 3072
    x_test = np.reshape(x_test, (x_test.shape[0], -1)) # 10000, 3072
    y_train = to_categorical(y_train, num_classes=10) # 50000, 10
    y_test = to_categorical(y_test, num_classes=10) # 10000, 10
    x_train = x_train/255
    x_test = x_test/255
    

    #Create the model
    model = BaselineModel()
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])    

    # Define loss and optimizer
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=2, validation_split=0.2)

    # Test trained model
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)    
    print(model.metrics_names)
    print(score)
