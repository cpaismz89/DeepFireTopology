# Run Keras on CPU
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = " "         # -1 if CPU

# Importations
from IPython.display import Image

# Compressed pickle
import pickle
from compress_pickle import dump as cdump
from compress_pickle import load as cload
import io

# Importations
import numpy as np
import pandas as pd
from time import time
import re
import os
import random
import time

# Deep learning
import tensorflow as tf
import keras 
from keras.models import Sequential, Model, load_model
from keras.regularizers import l2
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Image Processing
from imutils import paths, build_montages
import imutils
import cv2

# Gridsearch
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, LabelBinarizer, MultiLabelBinarizer, LabelEncoder
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, roc_auc_score, auc, confusion_matrix, accuracy_score, classification_report

# Visuals
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting training class
from IPython.display import clear_output

# Visuals scripts
import sys
sys.path.append('..')    # Parent folder

from drawer.keras_util import convert_drawer_model
from drawer.pptx_util import save_model_to_pptx
from drawer.matplotlib_util import save_model_to_file