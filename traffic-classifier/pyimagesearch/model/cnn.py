from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten,Dropout,MaxPooling2D,Merge,merge
from keras import optimizers

import csv

class cnn_net:
    @staticmethod
    
    # adapted from here http://people.idsia.ch/~juergen/nn2012traffic.pdf
    def idsia(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(nb_filter = 250,nb_row = 5, nb_col = 5,activation='relu',input_shape = (width ,height ,depth)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # second set of CONV => RELU => POOL
        model.add(Conv2D(nb_filter = 500,nb_row = 3, nb_col = 3,activation='relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        # third set of CONV => RELU => POOL
        model.add(Conv2D(nb_filter = 750,nb_row = 3, nb_col = 3,activation='relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(300,activation='relu'))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes,activation='softmax'))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
    
    def cnn_nomax(width, height, depth, classes, weightsPath=None):
        model = Sequential()
        model.add(Conv2D(nb_filter = 16,nb_row = 2, nb_col = 2,activation='relu',input_shape = (width ,height ,depth)))
        model.add(Conv2D(nb_filter = 48,nb_row = 2, nb_col = 2,activation='relu'))
        model.add(Conv2D(nb_filter = 96,nb_row = 2, nb_col = 2,activation='relu'))
        model.add(Conv2D(nb_filter = 128,nb_row = 2, nb_col = 2,activation='relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes,activation = 'softmax'))
        
                # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
    


    
