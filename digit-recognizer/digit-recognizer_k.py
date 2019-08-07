#!/usr/bin/python3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
#part of scipy
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv("~/Kaggle/digit-recognizer/train.csv")
test = pd.read_csv("~/Kaggle/digit-recognizer/test.csv")

X_train = (train.iloc[:,1:].values.astype('float32'))
y_train= (train.iloc[:,0].values.astype('int32'))
X_test = (test.values.astype('float32'))

X_train
y_train
