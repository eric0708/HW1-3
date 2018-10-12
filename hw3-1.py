import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt


def load_data():
	(x_train,y_train),(x_test,y_test)=mnist.load_data()
	number=10000
	x_train=x_train[0:number]
	y_train=y_train[0:number]
	x_train=x_train.reshape(number,28*28)
	x_test=x_test.reshape(x_test.shape[0],28*28)
	x_train=x_train.astype('float32')
	x_test=x_test.astype('float32')
	y_train=np_utils.to_categorical(y_train,10)
	y_test=np_utils.to_categorical(y_test,10)
	x_train=x_train
	x_test=x_test
	x_train=x_train/255
	x_test=x_test/255
	np.random.shuffle(y_train)
	return (x_train,y_train),(x_test,y_test)

epochnum=4000
(x_train,y_train),(x_test,y_test)=load_data()
trainingloss=np.zeros(epochnum)
testingloss=np.zeros(epochnum)
epochnumber=np.zeros(epochnum)


model=Sequential()
model.add(Dropout(0.2))
model.add(Dense(input_dim=28*28,units=1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])

for i in range(epochnum):
	history=model.fit(x_train,y_train,batch_size=100,epochs=1)
	result=model.evaluate(x_test,y_test)
	print('\nTest Loss:',result[0])
	trainingloss[i]=history.history['loss'][0]
	testingloss[i]=result[0]
	epochnumber[i]=i+1

plt.plot(epochnumber,trainingloss,color='blue')
plt.plot(epochnumber,testingloss,color='red')
plt.show()

