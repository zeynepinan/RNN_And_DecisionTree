import numpy as np
import pandas as pd

feature_names = ['region-centroid-col', 'region-centroid-row', 'region-pixel-count', 'short-line-density-5', 
                 'short-line-density-2', 'vedge-mean',  'vegde-sd', 'hedge-mean', 'hedge-sd', 'intensity-mean',
                 'rawred-mean','rawblue-mean','rawgreen-mean','. exred-mean','exblue-mean','exgreen-mean',
                 'value-mean','saturatoin-mean','hue-mean', 'class']
dataset=pd.read_csv("segment.dat", sep=' ' ,names=feature_names)
#print(data)
classdata=dataset.iloc[:,-1:].values
featuredata= dataset.iloc[:,:-1].values

from keras.utils import np_utils
classdata_utils=np_utils.to_categorical(classdata)
print(classdata_utils)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featuredata, classdata_utils, test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test)


rnnmodel = Sequential()

rnnmodel.add(SimpleRNN(50,activation='tanh' , return_sequences=True))
rnnmodel.add(Dropout(0.2))

rnnmodel.add(SimpleRNN(50,activation='tanh' , return_sequences=True))
rnnmodel.add(Dropout(0.2))

rnnmodel.add(SimpleRNN(50,activation='relu', return_sequences = True))
rnnmodel.add(Dropout(0.2))

rnnmodel.add(SimpleRNN(50,activation='relu', return_sequences = True))
rnnmodel.add(Dropout(0.2))

rnnmodel.add(SimpleRNN(50,activation='relu', return_sequences = True))
rnnmodel.add(Dropout(0.2))

rnnmodel.add(SimpleRNN(50, activation='relu'))
rnnmodel.add(Dropout(0.2))

# Adding the output layer
rnnmodel.add(Dense(8, activation='softmax'))

rnnmodel.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

egitim=rnnmodel.fit(X_train,y_train, epochs=50, batch_size=32,  verbose=1, validation_data=(X_test, y_test))

scores=rnnmodel.evaluate(X_train,y_train)
print("%s: %.2f%%" %(rnnmodel.metrics_names[1],scores[1]*100))

from matplotlib import pyplot as plt
# Plot training & validation accuracy values
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.plot(egitim.history['accuracy'])
plt.plot(egitim.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid()
# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(egitim.history['loss'])
plt.plot(egitim.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid()
plt.show()
