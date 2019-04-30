
# -*- coding: utf-8 -*-
"""

@author:Blesson George
"""

import keras,json
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import numpy as np

from samplex import balanced_sample_maker
from samplex import next_picker
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate
batch_size=4
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train.shape,x_test.shape,type(x_train),y_train.shape,y_test.shape)
data=np.concatenate((x_train,x_test),axis=0)
#data=data[0:100]
print('DATA',data.shape)


labels=np.concatenate((y_train,y_test),axis=0)
#labels=labels[0:100]
print('LABELS',labels.shape)


#z-score
mean = np.mean(data,axis=(0,1,2,3))
std = np.std(data,axis=(0,1,2,3))
data=(data-mean)/std+1e-7

num_classes = 10

weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(-1.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )





#Train trainset and labelset
trainset,labelset,sample_idx=balanced_sample_maker(data,labels,1)
newX = data[np.setdiff1d(np.arange(data.shape[0]), sample_idx)]
newy = labels[np.setdiff1d(np.arange(labels.shape[0]), sample_idx)]
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.save_weights('model.h5')
iter=0
results={}
for iter in range(0,1000):
    print('Iteration No',iter,'starts')    
    print('Trainset Shape',trainset.shape,'Trainlabels',labelset.shape)
    datagen.fit(trainset)
    label_one_hot=np_utils.to_categorical(labelset,num_classes)
    model.fit_generator(datagen.flow(trainset, label_one_hot, batch_size=4),\
                    steps_per_epoch=trainset.shape[0] //batch_size,epochs=125,\
                    verbose=2,callbacks=[LearningRateScheduler(lr_schedule)])#,EarlyStopping(monitor='val_loss', patience=2, verbose=2,mode='auto',baseline=None)])#,restore_best_weights=True)])
    #save to disk
    #model_json = model.to_json()
    #with open('model.json', 'w') as json_file:
     #   json_file.write(model_json)

    #model.save_weights('model.h5')  
    ans=model.predict(newX,batch_size=32, verbose=2)
    print('prediction',ans.shape,type(ans))  
    maxlabel=np.argmax(ans,axis=1)
    #maxlabel=np.array(maxlabel.reshape(maxlabel.shape[0],))
    print('MaxLabel',maxlabel.shape)
    maxval=np.max(ans,axis=1)
    print('Maxval',maxval.shape)
    newy=np.array(newy.reshape(newy.shape[0],))
    print('newy',newy.shape)
    print('maxval',maxval)
    print('maxlabel',maxlabel)
    print('newy',newy)

    incorrect=np.where(maxlabel!=newy)
    print(incorrect)
    newyset=np_utils.to_categorical(newy,num_classes)
    scores = model.evaluate(newX,newyset, batch_size=128, verbose=10)
    #print('\nIteration No:%i Test result: %.3f loss: %.3f' % (iter,scoresi[1]*100,scores[0]))
    results[iter]=('Accuracy:',scores[1]*100,'Loss:',scores[0],'Training Size:',trainset.shape[0])
    with open('resapr_30.json','w') as fp:
        json.dump(results,fp)   
    #    file.write(res)
    trainset,labelset,newX,newy=next_picker(newX,newy,trainset,labelset,incorrect,maxval)
    print('\nIteration No:%i Test result: %.3f loss: %.3f' % (iter,scores[1]*100,scores[0]))
    model.load_weights('model.h5')
"""
#    iter+=1
    for i in range(0,newX.shape[0]):
        if (np.argmax(model.predict(newX[i].reshape(1,32,32,3)),axis=1)!=newy[i]):

            incorrect.append(i)
        prob_list[i]=(np.max(model.predict(newX[i].reshape(1,32,32,3)),axis=1))
    print('Total No.',newX.shape[0],'InCorrect Predictions',len(incorrect))    
   ''' 
    newyset=np_utils.to_categorical(newy,num_classes)
    scores = model.evaluate(newX,newyset, batch_size=128, verbose=10)
    #print('\nIteration No:%i Test result: %.3f loss: %.3f' % (iter,scoresi[1]*100,scores[0]))
    results[iter]=('Accuracy:',scores[1]*100,'Loss:',scores[0],'Training Size:',trainset.shape[0])
    with open('resapr_26.json','w') as fp:
        json.dump(results,fp)   
    #    file.write(res)
    trainset,labelset,newX,newy=next_picker(newX,newy,trainset,labelset,incorrect,maxval)
    print('\nIteration No:%i Test result: %.3f loss: %.3f' % (iter,scores[1]*100,scores[0]))
    model.load_weights('model.h5')"""

