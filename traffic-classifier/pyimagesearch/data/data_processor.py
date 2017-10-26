
import pickle
import pandas as pd
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
import csv


class data_processor:
    @staticmethod

    def build(dataPath):
        training_file = dataPath + 'train.p'
        validation_file = dataPath + 'valid.p'
        testing_file = dataPath + 'test.p'

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        X_train, y_train = train['features'], train['labels']
        X_valid, y_valid = valid['features'], valid['labels']
        X_test, y_test = test['features'], test['labels']

        return X_train, y_train, X_valid, y_valid , X_test, y_test
    
    
    def label_names(label_file_name):
        # import label names for validation
        label_names_file = label_file_name
        label_names = pd.read_csv(label_names_file)
        
        return label_names
    
    def one_hot_encoding(labels,nb_classes):
        # adapted from here: https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
        targets = np.array(labels).reshape(-1)
        results = np.eye(nb_classes)[targets]
        return results
    
### image processing methods ###

    def convert_rgb_greyscale(rgb_dataset):
        greyscale_dataset = np.sum(rgb_dataset/3,axis = 3, keepdims=True)
        
        return greyscale_dataset
    
    # example : https://github.com/aleju/imgaug
    def augment_images(img_dataset,seed):
        ia.seed(seed)

        seq = iaa.Sequential([
            #iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True) # apply augmenters in random order
 
        return seq.augment_images(img_dataset)

    # applies Histograms Equalization
    # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    # inputs dataset needs to be samples,x,y,rgb
    def converted(img_dataset):
        img_yuv = cv2.cvtColor(img_dataset, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        return img_output
    
    # below does the following
    def pre_processing_single_img (img):

        img_y = cv2.cvtColor(img, (cv2.COLOR_BGR2YUV))[:,:,0]
        img_y = (img_y / 255.).astype(np.float32)
        #img_y = exposure.adjust_log(img_y)
        img_y = (exposure.equalize_adapthist(img_y,) - 0.5)
        img_y = img_y.reshape(img_y.shape + (1,))

        return img_y

    # borrowed from here: https://github.com/jokla/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb
    def contrast_limit_hist_equalize_images(img_dataset,greyscale = True):

            img_dateset_eq = np.empty((img_dataset.shape[0],img_dataset.shape[1],img_dataset.shape[2],1))
            if not greyscale:
                for i in range(len(img_dataset)):
                    img = img_dataset[i]

                    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
                    
                    # adapt history expects values between 0 and 1. This seems logical since not varying values
                    # https://www.mathworks.com/help/images/ref/adapthisteq.html
                    # normalize values by range. values would be from 0 to 1
                    img_y = (img_y / 255.0).astype(np.float32)
                    
                    #center at 0 and values range between -0.5 and 0.5
                    img_y = (exposure.equalize_adapthist(img_y,) - 0.5)
                    img_y = img_y.reshape(img_y.shape + (1,))
                    
                    

                    # equalize the histogram of the Y channel
                    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                    
                    # normalize values by range. values would be from 0 to 1

                    # convert the YUV image back to RGB format
                    #img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                    img_dateset_eq[i] = img_y
            else:
                equalizer = lambda x: cv2.equalizeHist(x)
                img_dataset_eq = np.array([equalizer(xi) for xi in img_dateset_eq])

                img_dataset_eq =  np.reshape(img_dataset_eq,
                                             (img_dataset.shape[0],
                                              img_dataset.shape[1],
                                              img_dataset.shape[2],
                                              img_dataset.shape[3])
                                            )


            return img_dateset_eq

    
    def hist_equalize_images(img_dataset,greyscale = True):
        
        img_dateset_eq = img_dataset
        if not greyscale:
            for i in range(len(img_dataset)):
                img = img_dateset_eq[i]

                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

                # equalize the histogram of the Y channel
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

                # convert the YUV image back to RGB format
                img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                img_dateset_eq[i] = img_output
        else:
            equalizer = lambda x: cv2.equalizeHist(x)
            img_dataset_eq = np.array([equalizer(xi) for xi in img_dateset_eq])

            img_dataset_eq =  np.reshape(img_dataset_eq,
                                         (img_dataset.shape[0],
                                          img_dataset.shape[1],
                                          img_dataset.shape[2],
                                          img_dataset.shape[3])
                                        )
            
        
        return img_dateset_eq
    
    def hist_equalize_images_old(img_dataset):
        
        equalizer = lambda x: cv2.equalizeHist(x)
        img_dataset_eq = np.array([equalizer(xi) for xi in img_dataset])

        img_dataset_eq =  np.reshape(img_dataset_eq,
                                     (img_dataset.shape[0],
                                      img_dataset.shape[1],
                                      img_dataset.shape[2],
                                      img_dataset.shape[3])
                                    )
        
        return img_dataset_eq
    
    def pre_process_image(img_dataset):

        img_dataset[:,:,0] = cv2.equalizeHist(img_dataset[:,:,0])
        img_dataset[:,:,1] = cv2.equalizeHist(img_dataset[:,:,1])
        img_dataset[:,:,2] = cv2.equalizeHist(img_dataset[:,:,2])
        img_dataset = img_dataset/255.-.5

        return image
    
    def normalize_images(img_dataset):
        #img_dataset_norm = (img_dataset - 128) / 128
        
        norm = img_dataset
        b=img_dataset[:,:,0]
        g=img_dataset[:,:,1]
        r=img_dataset[:,:,2]

        sum=b+g+r

        norm[:,:,0]=b/sum*255.0
        norm[:,:,1]=g/sum*255.0
        norm[:,:,2]=r/sum*255.0
        
        norm_rgb=cv2.convertScaleAbs(norm)
        
        return norm_rgb


    def explore_greyscale_black_images(grescale_dataset):
        img_contrast = np.empty([grescale_dataset.shape[0]])

        # 0 = black and 255 = white. so less than 128 then more black
        for i in range(len(grescale_dataset)):
            if np.sum(grescale_dataset[i]<128)/(32.0*32.0) > 0.6:
                img_contrast[i] = 1
            else:
                img_contrast[i] = 0   
                
        plt.hist(img_contrast,bins = 10)

