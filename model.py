import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import flowkit as fk
from tensorflow import keras
windowsize = 9
from tensorflow.keras import layers

from sklearn.utils import class_weight

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

csv_file_path = "DeepLearning_for_ImagingFlowCytometry\data\data.csv"
fcs_file_path = "DeepLearning_for_ImagingFlowCytometry\data/test_comp_example.fcs"
#fcs_file_path = "DeepLearning_for_ImagingFlowCytometry\data/HD00 CD46.fcs"
fcs_as_csv_path = "DeepLearning_for_ImagingFlowCytometry\data/test_comp_example.csv"
comp_file_path = "DeepLearning_for_ImagingFlowCytometry\data/comp_complete_example.csv"
blank = "DeepLearning_for_ImagingFlowCytometry\data/blank.csv"

sample = fk.Sample(
    fcs_path_or_data=fcs_file_path,
    compensation=comp_file_path,
    ignore_offset_error=True,
    cache_original_events = True
)

sample.subsample_events(50000)

xform = fk.transforms.LinearTransform(
    'lin',
    param_t=1000,
    param_a=0
)

sample.apply_transform(xform, 1)

sample.export(fcs_as_csv_path)

df = pd.read_csv(fcs_as_csv_path)

print(df.describe())
print(np.bincount(df['FSC-A']))
plt.hist(df['Ax488-A'], 100)
plt.xlabel("FSC-A")
plt.ylabel("Count")
plt.show()

dffeatures =  df.drop(columns=['Ax488-A', 'PE-A', 'PE-TR-A', 'PerCP-Cy55-A',
       'PE-Cy7-A', 'Ax647-A', 'Ax700-A', 'Ax750-A', 'PacBlu-A', 'Qdot525-A',
       'PacOrange-A', 'Qdot605-A', 'Qdot655-A', 'Qdot705-A', 'Time'])
       
       
dflabels =  df.drop(columns=['FSC-A', 'FSC-W', 'SSC-A', 'PE-A', 'PE-TR-A', 'PerCP-Cy55-A',
       'PE-Cy7-A', 'Ax647-A', 'Ax700-A', 'Ax750-A', 'PacBlu-A', 'Qdot525-A',
       'PacOrange-A', 'Qdot605-A', 'Qdot655-A', 'Qdot705-A', 'Time'])
 
stest_features = dffeatures.drop(labels = range(0,7483), axis = 0)

strain_features = dffeatures.drop(labels = range(7483,10703), axis = 0)


stest_labels = dflabels.drop(labels = range(0,7483), axis = 0)

strain_labels = dflabels.drop(labels = range(7483,10702), axis = 0)


(train_data, train_labels), (test_data, test_labels) = (strain_features.values, strain_labels), (stest_features.values, stest_labels)

mean = train_data.mean(axis=0)

std = train_data.std(axis=0)

train_data = (train_data - mean) / std

test_data = (test_data - mean) / std


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(3, activation = tf.nn.selu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(3, activation = tf.nn.relu),
        #keras.layers.Dense(6, activation = tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae'])
    
    return model

model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end = '')

EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)

history = model.fit(train_data, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])

print(history.history.keys())

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mae']))
    plt.plot(history.epoch, np.array(history.history['val_mae']))
    plt.legend()
    plt.ylim(plt.ylim())
    plt.show()

plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    
print("\nTesting set Mean abs error: " + str(mae))
print("\nTesting set Loss:" + str(loss))

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([800, 1000], [800, 1000])

plt.show()

error = test_predictions - test_labels.to_numpy().flatten()
plt.hist(error, bins = 50)
plt.xlabel("Predictions error")
_ = plt.ylabel("Count")
plt.show()
