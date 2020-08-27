
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import cv2
import csv
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Cropping2D, Dropout
import sklearn
import math
import os
import random
#import matplotlib.pyplot as plt

##
## Functions definition
##

def extractLogLines(dataPath):
    lines = []
    dirlist = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]   # Creates a vector containing the names of each training folder
    for folder in dirlist:                                                              # for every folder in the data set                           
        with open(dataPath + '/' + folder + '/driving_log.csv') as csvfile:             # read the driving log file
            reader = csv.reader(csvfile)
            next(reader, None)                                                          # skip the headers
            for line in reader:
                lines.append(line)                                                      # creates a vector containing the lines in ALL the csv files
    print("Lines: ",len(lines))
    return lines
            
def extractDataAndLabels(dataPath, lines,images, measurements):
    for line in lines:
        for i in range(3):
            measurement = float(line[3])                                           #Extract the stearing measurement which is in the 4th position of the csv file
            if measurement == 0 and random.randint(0,100) < 30:                
                continue   
            source_path = line[i]                        # extract the i-th argument of each line which represents the path of the center/left/right camera image 
            filename = source_path.split('/')[-3] + '/' + source_path.split('/')[-2] + '/' + source_path.split('/')[-1]     #Extract the image path from the line
            current_path = dataPath + '/IMG/' + filename                            #Attach center camera file name the the path in order to obtain the full path 
            image = cv2.imread(source_path)
            images.append(image)                                                        # Create the vectors containing the images
            if i == 1 : 
                measurement += 0.2                                                      # apply a corretion factor to adapt the steering angle for the left image
            elif i == 2: 
                measurement -= 0.2                                                      # apply a corretion factor to adapt the steering angle for the right image
            measurements.append(measurement)                                            # Create the vectors containing the stearing angles measurements 
            
def imagePreprocessing(model):
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))                 # normalize and mean center the data
    model.add(Cropping2D(cropping=((70,25),(0,0))))                                     # crop the upper part of the images which is not of our interest
    
def lenetArchitecture(model):
    model = Sequential()
    model.add(Convolution2D(6,5,5, activation = "relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation = "relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    
def nvidiaArchitecture(model):
    model.add(Convolution2D(24,5,5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(36,5,5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(48,5,5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(64,3,3, activation = "relu"))
    model.add(Convolution2D(64,3,3, activation = "relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Dense(50))    
    model.add(Dropout(.5))
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Dense(1))   
      
def generator(samples, path, batch_size=32):
    num_samples = len(samples)
    while 1:                                                        # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []   
            
            extractDataAndLabels(path, batch_samples, images, measurements)
   
            #PROBLEM: the model is trained with data coming from a counterclock wise laps. This means that for most of the time the car is steering left.
            #        It results that the final model predicts to steer left even when it is better to go straight.
            #SOLUTION: we add to the training set the images flipped along the vertical axes (as if the curvature is on the opposite side).
            #       The corresponding steering angles are evaluated subtracting -1 to the orginal value.
            
            augmented_images, augmented_measurements = [], []
            
            for image, measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)

##
## Program begin             
##            

path = './data_new'                                                             # path where all the dataset is located

lines = extractLogLines(path)

from sklearn.model_selection import train_test_split                
train_samples, validation_samples = train_test_split(lines, test_size=0.2)      # randomly split the dataset into training and valitation set

batch_size=32                                                                   # Set our batch size

# compile and train the model using the generator function
train_generator = generator(train_samples, path, batch_size=batch_size)
validation_generator = generator(validation_samples, path, batch_size=batch_size)


#Create a model that, given a camera image, predicts the steering angle to apply to the car.

model = Sequential()
imagePreprocessing(model)
nvidiaArchitecture(model)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), \
            epochs=3, verbose=1)

model.save('model.h5')