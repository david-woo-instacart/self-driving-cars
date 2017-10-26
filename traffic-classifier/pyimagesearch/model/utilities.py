
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as sk

# Goal of this class is to store all diagnostics for models. i.e post eval of model
class utilities:
    
    @staticmethod
    
    def plt_accuracy(model_training_history):
        
        plt.plot(model_training_history.history['acc'])
        plt.plot(model_training_history.history['val_acc'])
        plt.legend(['train','validation'])
        plt.show()
    
    # this could be usefu
   # def plt_confusion_matrix():
        
    
    
    def get_label_names(y_classes,labels,label_class_id_col_name,label_name_col_name):
        
        df = pd.DataFrame({label_class_id_col_name:y_classes.ravel()})
        combine_df = pd.merge(df,labels, on = label_class_id_col_name)
        combine_np = combine_df[label_name_col_name].as_matrix()
        
        return combine_np
        
    # labels organized as class_id, sign_names
    # y hat = 1-D array of class_id
    # y_actual = 1-D array of class_id
    def get_prediction_classes(y_hat,y_actual,labels,label_class_id_col_name,label_name_col_name):
        
        y_hat_labels = utilities.get_label_names(y_hat,labels)
        y_actual_labels = utilities.get_label_names(y_actual,labels)
        
        return y_hat_labels,y_actual_labels
    
    # adapted from here http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def plt_confusion_matrix(y_hat_labels,y_test_labels,classes,normalize = False,figsize =16):

        classes = classes
        cmap=plt.cm.Blues

        cm = sk.confusion_matrix(y_hat_labels,y_test_labels)

        normalize = True
        if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(16,16), tight_layout={'h_pad':4})
        plt.imshow(cm, interpolation='nearest',cmap = cmap)

        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        plt.show()
    
    def get_prediction_matrix(dataset_model,y_test,model):
        
        y_hat_test = model.predict(dataset_model)
        
        #2. Get the class of each prediction and the probability
        sort_prediction = lambda x: np.argsort(x)[-1:]

        # this returns the sign
        y_hat_test_filtered = np.array([sort_prediction(xi) for xi in y_hat_test])

        y_hat_test_filtered_prob = np.empty((y_hat_test_filtered.shape[0],4))

        for i in range(len(y_hat_test_filtered)):
            # order is : image index, prediction class, actual class, predicted probability
            y_hat_test_filtered_prob[i] = [i,y_hat_test_filtered[i],y_test[i],y_hat_test[i][y_hat_test_filtered[i]]]
        
        # get correct classes
        correct_classes = y_hat_test_filtered_prob[np.where(y_hat_test_filtered_prob[:,1] == y_hat_test_filtered_prob[:,2])]
        
        # get wrong classes
        wrong_classes = y_hat_test_filtered_prob[np.where(y_hat_test_filtered_prob[:,1] != y_hat_test_filtered_prob[:,2])]
        
        return correct_classes, wrong_classes
    
    # label names has to be a pandas dataframe
    def plt_prediction_matrix(classes,label_names,dataset_orig,dataset_model):
        
        fig2 = plt.figure(figsize=(32,32), tight_layout={'h_pad':4})
        fig4 = plt.figure(figsize=(32,32), tight_layout={'h_pad':4})

        samples = 10

        # next sort wrong classes
        for i in range(samples):
            ax2 = fig2.add_subplot(samples,samples,i+1)
            ax4 = fig4.add_subplot(samples,samples,i+1)

            print('prob % class predicted %',
                  classes[i][3],
                  label_names.query('ClassId ==' + str(classes[i][1])),
                  label_names.query('ClassId ==' + str(classes[i][2])),
                 )
            ax2.imshow(dataset_orig[int(classes[i][0])])
            ax4.imshow(dataset_model[int(classes[i][0])].squeeze())
            
            
            
            
        classes = wrong_classes
        fig2 = plt.figure(figsize=(32,32), tight_layout={'h_pad':4})
        fig4 = plt.figure(figsize=(32,32), tight_layout={'h_pad':4})
        fig5 = plt.figure(figsize=(32,32), tight_layout={'h_pad':4})

        samples = 10

        # next sort wrong classes
        for j in range(samples):
            i = random.randint(1,100)
            ax2 = fig2.add_subplot(samples,samples,j+1)
            ax4 = fig4.add_subplot(samples,samples,j+1)
            ax5 = fig5.add_subplot(samples,samples,j+1)

            print('prob % class predicted %',
                  correct_classes[i][3],
                  signs.query('ClassId ==' + str(classes[i][1])),
                  signs.query('ClassId ==' + str(classes[i][2])),
                 )
            ax2.imshow(X_test[int(classes[i][0])])
            ax4.imshow(X_test_grey_aug[int(classes[i][0])].squeeze())
            ax5.imshow(X_test_eq[int(classes[i][0])].squeeze())
            
        plt.show()
        
