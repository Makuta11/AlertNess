########################################################################################
###############                          Imports                         ###############
########################################################################################
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.estimator import DNNClassifier
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import regularizers
from sklearn.model_selection import KFold

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

########################################################################################
###############                        Load Data                         ###############
########################################################################################
data = pd.read_csv('dataMATLAB/features_allrecordings.csv')

# Find bands of the 2 frontal electrodes
bands = []
for name in data.columns:
  if "T8" in name:
    continue 
  elif "T7" in name:
    continue
  else:
    bands.append(name)

# Load data to GPU
data2 = data[(bands)] 
print(f'the shape of the data is: {data2.shape}\n')

# Create observation and target data
X = data2.drop(['Class'], axis=1).values
y = data2['Class']

# create masks for setting labels
alert_mask = (y=="Alert")
neutral_mask = (y=="Neutral")
drowsy_mask = (y=="Drowsy")

# reindex targets
y.loc[alert_mask] = int(0)
y.loc[neutral_mask] = int(1)
y.loc[drowsy_mask] = int(2)

########################################################################################
###############    Function to Create The Feed Forward Neural Network    ###############
########################################################################################
def create_model(DROP_OUT=0.02):
  model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(24,)),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=64, activation='relu'), #kernel_regularizer='l2'),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=256, activation='relu'),#, kernel_regularizer='l2'),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=512, activation='relu'),# kernel_regularizer='l2'),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=1024, activation='relu'),# kernel_regularizer='l2'),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=1024, activation='relu'),# kernel_regularizer='l2'),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=512, activation='relu'),# kernel_regularizer='l2'),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=256, activation='relu'),# kernel_regularizer='l2'),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(DROP_OUT),
      keras.layers.Dense(units=3, activation='softmax'),# kernel_regularizer='l2')
  ])
  return model

########################################################################################
###############      Perform Cross Validation Training of the Model      ###############
########################################################################################
# Get split index for CV
kf = KFold(n_splits = 4, shuffle=True, random_state=42)
numSplits = kf.get_n_splits(X)

# Training Parameters
epochs = 1
batch_size = 32 
#callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Predefine variable
historyDic = dict()
modelDic = dict()
k = 0

for train_index, test_index in kf.split(X):
    # Keep up with Fold
    print(f'Currently running fold: {k+1}/{numSplits}')

    # Create model
    model = create_model(DROP_OUT=0.02)

    # Compile model
    model.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
                loss=tf.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    # Convert splits to tensors
    X_train = tf.convert_to_tensor(X[train_index])
    y_train = tf.convert_to_tensor([y[train_index]])
    X_test = tf.convert_to_tensor(X[test_index])
    y_test = tf.convert_to_tensor([y[test_index]])

    # Print size of data
    print(f'Size of training data: {X_train.shape}')
    print(f'Size of training labels: {y_train.shape}')


    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,tf.one_hot(y_train[0], depth=3))) \
        .batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test,tf.one_hot(y_test[0], depth=3))) \
        .batch(batch_size)

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset
        #callbacks=[callback]
    )

    # Append model results to dictionaries
    historyDic[k] = history
    modelDic[k] = model

    # Itterate k to save models
    k += 1

########################################################################################
###############          Plot Training/Validation Loss function          ###############
########################################################################################
# Plot confusion matrix
mpl.style.use('fivethirtyeight')

for k in range(4):
  # Print learning curves
  acc = historyDic[k].history['accuracy']
  val_acc = historyDic[k].history['val_accuracy']
  loss = historyDic[k].history['loss']
  val_loss = historyDic[k].history['val_loss']

  # Plot stlye
  mpl.style.use('fivethirtyeight')
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  #plt.xlim([900,1000])
  #plt.ylim([0.7,0.9])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  #plt.ylim([-5,200])
  plt.yscale('log')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')

  plt.savefig(f'Figs/TrainingLoss_2feat_CV_Fold{k+1}.png', dpi=300)

########################################################################################
###############                  Plot Confusion Matrix                   ###############
########################################################################################
k = 0;
f1 = dict()
for train_index, test_index in kf.split(X):
  ### CLASSIFIER SETUP ###
  # Run classifier on test data and create prediction label
  y_pred = np.argmax(modelDic[k].predict(X[test_index]), axis=1).astype(int)

  # Calculate accuracy by creatig confusion matrix
  cm = confusion_matrix(np.transpose(y[test_index].astype(int)), y_pred)
  f1[k] = f1_score(np.transpose(y[test_index].astype(int)), y_pred, average='macro')
  accuracy = float(cm.diagonal().sum())/np.transpose(y[test_index]).shape[0]
  print(f"\nAccuracy of FFNN For The Given Dataset : {round(accuracy*100, 2)}%")
  print(f"F-1 score of FFNN For The Given Dataset : {round(f1[k]*100, 2)}")

  conf_arr = cm
  sum = conf_arr.sum()
  conf_arr = 3 * conf_arr * 100.0 / ( 1.0 * sum )

  df_cm = pd.DataFrame(conf_arr, 
    index = [ 'Alert', 'Neutral','Drowsy'],
    columns = ['Alert', 'Neutral','Drowsy'])

  fig = plt.figure(figsize=(12,8))
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  cmap = sns.cubehelix_palette(light=1, as_cmap=True)
  res = sns.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0, fmt='.2f', cmap=cmap)
  plt.yticks([0.5,1.5,2.5], [ 'Alert', 'Neutral','Drowsy'],va='center')
  plt.title(f'Confusion Matrix - Fold{k+1} - F1:{round(f1[k]*100, 2)}')
  plt.savefig(f'Figs/Qmatrix_Fold{k}.png', dpi=300)

  k += 1

########################################################################################
###############                    Convert to TFLite                     ###############
########################################################################################
k = 0
for k in range(4):
    # Convert Keras model to TF Lite format.
    converter = tf.lite.TFLiteConverter.from_keras_model(modelDic[k])
    tflite_float_model = converter.convert()

    # Re-convert the model to TF Lite using quantization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    # Save the quantized model to file to the Downloads directory
    f = open(f'Models/kerasModel_2feat_Fold{k+1}_F1:{round(f1[k]*100,2)}.tflite', "wb")
    f.write(tflite_quantized_model)
    f.close()