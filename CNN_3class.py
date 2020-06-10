import tensorflow as tf
import os
import cv2
import glob
from time import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


def assign_label(img, typeo):
    return typeo


def loadfiles(typeo, img_dir, x, z):
    data_path = os.path.join(img_dir, '*.jpg')
    # print(data_path)
    files = glob.glob(data_path)
    for f1 in files:
        # print(f1)
        img = cv2.imread(f1, cv2.IMREAD_COLOR)

        label = assign_label(img, typeo)
        # X.append(img)
        # Z.append(str(label))
        x.append(img)
        z.append(label)


def loadset(which, lr, set):
    data_path = set + '/'+ lr + '/' + which + '/'
    x = []
    z = []
    loadfiles(0, data_path + 'Centre/', x, z)
    loadfiles(1, data_path + 'Right/', x, z)
    loadfiles(2, data_path + 'Left/', x, z)
    return x, z


def getlrset(lr, set):
    train_x1, train_y1 = loadset('train', lr, set)
    val_x1, val_y1 = loadset('val', lr, set)
    test_x1, test_y1 = loadset('test', lr, set)

    train_x1 = np.array(train_x1)
    train_y1 = np.array(train_y1)
    train_x1 = train_x1 / 255

    val_x1 = np.array(val_x1)
    val_y1 = np.array(val_y1)
    val_x1 = val_x1 / 255

    test_x1 = np.array(test_x1)
    test_y1 = np.array(test_y1)
    test_x1 = test_x1 / 255
    return train_x1, train_y1, val_x1, val_y1, test_x1, test_y1


def makemodel():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(42, 50, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3, activation="softmax"))
    return model


model = makemodel()
model2 = makemodel()

model.summary()
whichset = int(input('Dataset? 0:Chimera. 1:Camera \n'))
lr = 'left'
if whichset == 0:
    set1 = 'Output_haar'
else:
    set1 = 'camera_out'
to_save = int(input('Save models? 0:Yes. 1:No \n'))
if to_save == 0:
    saving = True
else:
    saving = False
train_x, train_y, val_x, val_y, test_x, test_y = getlrset(lr, set1)


print('Train set size:' + str(len(train_x)) + '\n')
print('Val set size:' + str(len(val_x)) + '\n')
print('Test set size:' + str(len(test_x)) + '\n')

es1 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


if set1 == 'camera_out':
    logdir1 = 'camera_left_logs/{}'
    logdir2 = 'camera_right_logs/{}'
    tensorboard1 = TensorBoard(log_dir=logdir1.format(time()))
    tensorboard2 = TensorBoard(log_dir=logdir2.format(time()))
    ckpt1 = ModelCheckpoint(filepath='best_model_ckpt_left_camera.hdf5', monitor='val_accuracy', verbose=1,
                            save_best_only=True,
                            mode='max')
    ckpt2 = ModelCheckpoint(filepath='best_model_ckpt_right_camera.hdf5', monitor='val_accuracy', verbose=1,
                            save_best_only=True,
                            mode='max')
elif set1 == 'Output_haar':
    logdir1 = 'Chimera_left_logs/{}'
    logdir2 = 'Chimera_right_logs/{}'
    tensorboard1 = TensorBoard(log_dir=logdir1.format(time()))
    tensorboard2 = TensorBoard(log_dir=logdir2.format(time()))
    ckpt1 = ModelCheckpoint(filepath='best_model_ckpt_left.hdf5', monitor='val_accuracy', verbose=1,
                            save_best_only=True,
                            mode='max')
    ckpt2 = ModelCheckpoint(filepath='best_model_ckpt_right.hdf5', monitor='val_accuracy', verbose=1,
                            save_best_only=True,
                            mode='max')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=200, batch_size=10, validation_data=(val_x, val_y),
                    callbacks=[es1, ckpt1, tensorboard1])
test1 = np.argmax(model.predict(test_x), axis=-1)

if saving:
    if set1 == 'camera_out':
        model.save('3classmodel_left_camera.h5')
    else:
        model.save('3classmodel_left.h5')
print(test1)

print('\n Model 1 stopped \n')

lr = 'right'
train_x, train_y, val_x, val_y, test_x, test_y = getlrset(lr, set1)

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(train_x, train_y, epochs=200, batch_size=10, validation_data=(val_x, val_y),
                      callbacks=[es2, ckpt2, tensorboard2])
test1 = np.argmax(model2.predict(test_x), axis=-1)
print(test1)
if saving:
    if set1 == 'camera_out':
        model.save('3classmodel_right_camera.h5')
    else:
        model.save('3classmodel_right.h5')

