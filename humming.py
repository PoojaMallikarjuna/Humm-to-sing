import io
import librosa
from librosa import display
from librosa.effects import trim
import matplotlib.pyplot as plt
from IPython.display import Audio
#from pyAudioProcessing import clean
from matplotlib.pyplot import specgram
import soundfile as sf
import numpy as np
import glob
import os
import pandas as pd
os.chdir("/content/drive/My Drive")
!ls

for path in file_paths:
  path = path.strip()
  humming = source+path

def feature_extraction(file_name):
  X, sample_rate = librosa.load(file_name)
  if X.ndim > 1:
    X = X[:, 0]
  Y = X.T
  
  X=trim(Y,top_db=60)
  A = np.array(X[1],dtype=float)
  stft = np.abs(librosa.stft(A))
  mfccs = np.mean(librosa.feature.mfcc(y=A, sr=sample_rate, n_mfcc=40).T, axis=0)
  #print(mfccs)
  mel = np.mean(librosa.feature.melspectrogram(A, sr=sample_rate).T, axis=0)
  #print(mel)
  spectral_centroids = np.mean(librosa.feature.spectral_centroid(A,sr=sample_rate)[0])
  spectral_bandwidth_2 = np.mean(librosa.feature.spectral_bandwidth(A+0.01,sr=sample_rate)[0])
  spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(A+0.01,sr=sample_rate)[0])
  return mfccs, mel, spectral_centroids, spectral_bandwidth_2, spectral_rolloff
  
  def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):  ## .wav audio format
  features, labels = np.empty((0, 171)), np.empty(0)  # 193 features total. This can vary

  for label, sub_dir in enumerate(sub_dirs):  ##Enumerate() function adds a counter to an iterable.
      for file_name in glob.glob(os.path.join(parent_dir, sub_dir,file_ext)):  ##parent is audio-data, sub_dirs are audio classes
          try:
              mfccs, mel, spectral_centroids, spectral_bandwidth_2, spectral_rolloff = feature_extraction(file_name)
          except Exception as e:
              print("[Error] there was an error in feature extraction. %s" % (e))
              continue

          extracted_features = np.hstack([mfccs, mel, spectral_centroids, spectral_bandwidth_2, spectral_rolloff])  # Stack arrays in sequence horizontally (column wise)
          features = np.vstack([features, extracted_features])  # Stack arrays in sequence vertically (row wise).
          labels = np.append(labels, label)
      print("Extracted features from %s, done" % (sub_dir))
  return np.array(features), np.array(labels, dtype=np.int)
  
audio_directories = os.listdir("audio dataset/")
audio_directories.sort()

features, labels = parse_audio_files('audio dataset', audio_directories) #(parent dir,sub dirs)
#print(features)
np.save('feat.npy', features) ##NumPy array file created. Files are binary files to store numpy arrays
np.save('label.npy', labels)

abels = np.load('label.npy') # 10 labels total
#print(labels)

# For future label de-encoding
label_classes = np.array(['Agar Tum','heeguirabahude','karabu','kolaveri','naguvanayana','neeparichaya','thumkyu','zara'])
print(label_classes)

features= np.load('feat.npy')
print(len(features))
print(features)

df = pd.DataFrame(features)

# Add a new column for class (label), this is our target
df['Audio class'] = pd.Categorical.from_codes(labels, label_classes)

df[[0,1,2,3,4,5,6,7,'Audio class']]
df

import numpy as np
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

#Load data from generated numpy files
X = np.load('feat.npy') # list of features
y = np.load('label.npy').ravel() # labels are the target

# Split into train and test sets (400 Audios total)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Data scaling (NOT IMPLEMENTING)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# Implement simple linear SVM
svm_clf = SVC(C=28.0, gamma = 0.00001, decision_function_shape="ovr") #These parameters can be modified

# Fit model
svm_clf.fit(X_train, y_train) #From Beif github
svm_clf.fit(X_train_scaled, y_train) # HandsOn book

# Make predictions
y_pred = svm_clf.predict(X_train_scaled)
y_predict = svm_clf.predict(X_test)

print('Prediction')
print(y_predict)
#print
print("Actual")
print(y_test)

# Accuracy
acc = svm_clf.score(X_test, y_test)
print
print("accuracy=%0.3f" %acc)

import numpy as np
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Load data 
X = np.load("feat.npy")
y = np.load('label.npy').ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 233)

# Neural Network Construction
model = Sequential()

# Architecture
model.add(Conv1D(64, 3, activation='relu', input_shape = (171, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Which is the best loss function for binary (multiple) classification
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Convert label to onehot
y_train = keras.utils.np_utils.to_categorical(y_train - 1, num_classes=10) # Converts a class vector (integers) to binary class matrix
y_test = keras.utils.np_utils.to_categorical(y_test - 1, num_classes=10)

X_train = np.expand_dims(X_train, axis=2) # Make 2-dim into 3-dim array to fit model
X_test = np.expand_dims(X_test, axis=2)

# Train Network
model.fit(X_train, y_train, batch_size=64, epochs=200)

# Compute accuracy with test data
score, acc = model.evaluate(X_test, y_test, batch_size=16) # Computes the loss & accuracy based on the input you pass it

print('Test score:', score) #loss
print('Test accuracy:', acc)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd 

#Load data 
X = np.load('feat.npy') 
y = np.load('label.npy').ravel() 

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize classifier
gnb_clf= GaussianNB() #check input params

# Train model
gnb_clf.fit(X_train, y_train)
#model = gnb_clf.fit(X_train, y_train)

# Make predictions
prediction = gnb_clf.predict(X_test)

print('Predicted values')
print(prediction)
print
print('Actual values')
print(y_test)
print

# Evaluate accuracy
#Similar ways to do it
#print(accuracy_score(y_test,prediction)) 
print
acc = gnb_clf.score(X_test, y_test) 
print("Accuracy = %0.3f" %acc)

from sklearn.ensemble import RandomForestClassifier #Random Forest classifier
import pandas as pd 
import numpy as np
np.random.seed(0)
X = np.load('feat.npy')
y = np.load('label.npy').ravel() 
  
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  
# Initialize classifier
rf_clf = RandomForestClassifier(n_jobs=2, random_state=0) #Check params
  
# Train model
rf_clf.fit(X_train, y_train)
  
# Make predictions
y_prediction = rf_clf.predict(X_test)
print('Predicted values')
print(y_prediction)
#print
print('Actual values')
print(y_test)
#print
  
# Evaluate accuracy
acc = rf_clf.score(X_test, y_test)
print("Accuracy = %0.3f" %acc)
rf_clf.predict_proba(X_test)[0:10]

prediction_decoded = label_classes[y_prediction]
actual_value_decoded = label_classes[y_test]
#print(y_prediction)
#print(y_test)
print('Prediction decoded')
print(prediction_decoded)
print('Actual class decoded')
print(actual_value_decoded)

pd.crosstab(actual_value_decoded, prediction_decoded)

import os
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow

# Load data 
X = np.load("feat.npy")
y = np.load('label.npy').ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 233)

#batch_size = 35
# nb_epochs = 400

# Reshape data for LSTM (Samples, Timesteps, Features)
X_train = np.expand_dims(X_train, axis=2) #(280,193,1)
X_test = np.expand_dims(X_test, axis=2)

y_train = tensorflow.keras.utils.to_categorical(y_train - 1, num_classes=10) # Converts a class vector (integers) to binary class matrix
y_test = tensorflow.keras.utils.to_categorical(y_test - 1, num_classes=10)

# Build RNN Neural Network
print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=X_train.shape[1:]))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(y_train.shape[1], activation='softmax'))
#model.add(Dense(10, activation='sigmoid'))

#model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
#model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
#model.add(Dense(units=genre_features.train_Y.shape[1], activation='softmax'))
          
print("Compiling ...")
model.compile(loss='categorical_crossentropy', # for multiple classes
              optimizer='adam', 
              metrics=['accuracy'])

print(model.summary())

print("Training ...")
model.fit(X_train, y_train, batch_size=35, epochs=100)

print("\nValidating ...")
score, accuracy = model.evaluate(X_test, y_test, batch_size=35, verbose=1)
print("Loss:  ", score)
print("Accuracy:  ", accuracy)



