import numpy as np
import tensorflow as tf
import cv2
import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
import re
import pickle


keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)

def dropout_layer_factory():
    return Dropout(rate=0.2, name='dropout')

class HandShapeFeatureExtractor:


    def __init__(self):
        real_model = load_model(os.path.join('./', 'aug_10k4.h5'))
        self.model = real_model


    def pre_process_input_image(self,crop):
        try:
            img = cv2.resize(crop, (200, 200))
            img_arr = np.array(img) / 255.0
            img_arr = img_arr.reshape(1, 200, 200, 1)
            return img_arr
        except Exception as e:
            print(str(e))
            raise

    # calculating dimensions f0r the cropping the specific hand parts
    # Need to change constant 80 based on the video dimensions
    @staticmethod
    def bound_box(x, y, max_y, max_x):
        y1 = y + 80
        y2 = y - 80
        x1 = x + 80
        x2 = x - 80
        if max_y < y1:
            y1 = max_y
        if y - 80 < 0:
            y2 = 0
        if x + 80 > max_x:
            x1 = max_x
        if x - 80 < 0:
            x2 = 0
        return y1, y2, x1, x2

    def extract_feature(self, image):
        try:
            img_arr = self.__pre_process_input_image(image)
            # input = tf.keras.Input(tensor=image)
            return self.model.predict(img_arr)
        except Exception as e:
            raise
def one_hot_encoder(x):
    b = np.zeros((x.size, x.max()+2))
    b[np.arange(x.size),x] = 1
    return b

def main():

    alpha_dict = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    source_path = 'C:/Users/alchu/Pictures/hand_data'
    batch_size=32
    for j in [4]:
    #for j in range(1,6):
        hse = HandShapeFeatureExtractor()
        hse.model.summary()
        fc1 = hse.model.layers[-1]
        fc2 = hse.model.layers[-2]
        dropout1 = Dropout(0.5)
        x = dropout1(fc2.output)
        predictors = fc1(x)
        predictors1 = keras.activations.softmax(predictors)
        hand_net = Model(inputs=hse.model.input, outputs=predictors1)
        hand_net.summary()
        print('Files ending with {} taken as validation set'.format(j))
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        for i in os.listdir(source_path):
            img = cv2.imread(os.path.join(source_path,i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (200, 200))
            #img_arr = 1.0*cv2.equalizeHist(np.array(img)) / 255.0
            img_arr = np.array(img) / 255.
            #img_arr -= np.mean(img_arr)
            #img_arr /= np.std(img_arr)
            img_arr = img_arr.reshape(1, 200, 200, 1)
            if(int(i.split('.')[0][-1])==j):
                x_valid.append(img_arr)
                y_valid.append(alpha_dict.index(i.split("_")[0]))
            else:
                x_train.append(img_arr)
                y_train.append(alpha_dict.index(i.split("_")[0]))
        x_train = np.concatenate(x_train)
        y_train = np.array(y_train)
        x_valid = np.concatenate(x_valid)
        y_valid = np.array(y_valid)
        y_train = one_hot_encoder(y_train)
        y_valid = one_hot_encoder(y_valid)
        print(x_train.shape)
        print(y_train.shape)
        print(x_valid.shape)
        print(y_valid.shape)
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=10,
            height_shift_range=10,
            horizontal_flip=False)
        #train_datagen = ImageDataGenerator()
        val_datagen = ImageDataGenerator()
        train_data_generator = train_datagen.flow(x_train,y_train, batch_size=batch_size)
        val_data_generator = val_datagen.flow(x_valid,y_valid, batch_size=batch_size)
        hand_net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min',patience=50)
        mc = ModelCheckpoint('./checkpoints/aug_softmax_10k{}.h5'.format(j), monitor='val_accuracy', mode='max', verbose=1,save_best_only=True)
        csv_logger = CSVLogger('./checkpoints/aug_softmax_10k{}.csv'.format(j), separator=",", append=False)
        history = hand_net.fit(train_data_generator,epochs = 10000,verbose = 1,steps_per_epoch=len(x_train)/batch_size,validation_data = val_data_generator,validation_steps=len(x_valid)/batch_size,shuffle = True,callbacks=[mc,csv_logger])


	# hand_net.save('./50_epoch.h5')
	# with open('./history_saved.pkl','wb') as output:
	# 	pickle.dump(history,output)

main()
