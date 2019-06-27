# coding:utf-8
# coding: utf-8
# *****Main Trainning*****
# *****修改的一个现成的代码*****
import os
import numpy as np
from PIL import Image
from keras import callbacks
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf

# Pre process images
from tensorflow.python.client import session


class PreFile(object):
    def __init__(self, FilePath, ImageType):
        self.FilePath = FilePath

        self.ImageType = ImageType

    def FileReName(self):
        count = 0
        for type in self.ImageType:
            subfolder = os.listdir(self.FilePath + type)
            for subclass in subfolder:
                print(subclass)
                print(self.FilePath + type + '/' + subclass)
                os.rename(self.FilePath + type + '/' + subclass,
                          self.FilePath + type + '/' + str(count) + '_' + subclass.split('.')[0] + ".jpg")
            count += 1

    def FileResize(self, Width, Height, Output_folder):
        for type in self.ImageType:
            print(type)
            files = os.listdir(self.FilePath + type)
            for i in files:
                img_open = Image.open(self.FilePath + type + '/' + i)
                conv_RGB = img_open.convert('RGB')
                new_img = conv_RGB.resize((Width, Height), Image.BILINEAR)
                new_img.save(os.path.join(Output_folder, os.path.basename(i)))


# 主程序
class Training(object):
    def __init__(self, batch_size, number_batch, categories, train_folder):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.categories = categories
        self.train_folder = train_folder

    def read_train_images(self, filename):
        img = Image.open(self.train_folder + filename)
        return np.array(img)

    def train(self):
        train_img_list = []
        train_label_list = []
        for file in os.listdir(self.train_folder):
            files_img_in_array = self.read_train_images(filename=file)
            train_img_list.append(files_img_in_array)
            train_label_list.append(int(file.split('_')[0]))

        train_img_list = np.array(train_img_list)
        train_label_list = np.array(train_label_list)

        train_label_list = np_utils.to_categorical(train_label_list,
                                                   self.categories)

        train_img_list = train_img_list.astype('float32')
        train_img_list /= 255

        # 建立CNN
        model = Sequential()

        model.add(Convolution2D(
            filters=32,
            kernel_size=(5, 5),
            padding='same',
            input_shape=(100, 100, 3),
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        ))
        
        model.add(Convolution2D(
            filters=64,
            kernel_size=(2, 2),
            padding='same',
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        ))

        # Fully connected Layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))

        # model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(Activation('relu'))

        # model.add(Dropout(0.3))
        model.add(Dense(256))
        model.add(Activation('relu'))

        # model.add(Dropout(0.5))
        model.add(Dense(self.categories))
        model.add(Activation('softmax'))
        # Define Optimizer
        adam = Adam(lr=0.0001)
        # Compile the model
        model.compile(optimizer=adam,
                      loss="categorical_crossentropy",
                      metrics=['accuracy']
                      )
        # Fire up the network
        tbCallbacks = callbacks.TensorBoard(log_dir='logs/', histogram_freq=1, write_graph=True, write_images=True)
        history = model.fit(
            train_img_list,
            train_label_list,
            validation_split=0.15,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[tbCallbacks]
        )
        # SAVE your work model
        model.save('./modelname.h5')

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.grid(True)
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.grid(True)
        plt.legend()

        plt.show()


def MAIN():
    ImageType = ['type1', 'type2','type3', 'type4', 'type5', 'type6']

    FILE = PreFile(FilePath='Raw_Img/', ImageType=ImageType)

    FILE.FileReName()
    FILE.FileResize(Height=100, Width=100, Output_folder='train_img/')

    # Trainning Neural Network
    Train = Training(batch_size=10, number_batch=2, categories=6, train_folder='train_img/') #参数自己微调
    Train.train()


if __name__ == "__main__":
    MAIN()
