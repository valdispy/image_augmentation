# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:55:53 2019

@author: valdis
"""
#import decays
from re import findall
from tensorflow.keras.optimizers import Adam
import tensorflow as tf, glob

EPOCHS=30; BASE_RATE=1e-3
CROP_HEIGHT=200; CROP_WIDTH=300; CHANNELS=3
BATCH_SIZE=8; BUFFER_SIZE=4

def _parse_features(example_item):
    features = {'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string)}
    example_item = tf.io.parse_single_example(example_item, features)
    image = tf.image.decode_jpeg(example_item['image'], channels=CHANNELS)/255
    label = tf.image.decode_jpeg(example_item['label'], channels=CHANNELS)/255           
    return image, label

def _crop_image(image, label):   
    image_tensor = tf.stack([image, label], axis=0)
    combined_images = tf.image.random_crop(image_tensor, size = [2, CROP_HEIGHT, CROP_WIDTH, CHANNELS])
    return tf.unstack(combined_images, num=2, axis=0)    

def _random_hue(image, label):
    return tf.image.random_hue(image, max_delta=0.3), label

def _random_brightness(image, label):     
    return tf.image.random_brightness(image, max_delta=0.3), label
    
def _prepare_datasets(file_list):
    image_dataset = tf.data.TFRecordDataset(file_list)
    return image_dataset.map(_parse_features).map(_crop_image)\
                        .map(_random_brightness).map(_random_hue)\
                        .batch(batch_size=BATCH_SIZE).cache()\
                        .shuffle(buffer_size=BUFFER_SIZE).repeat()\
                        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def create_model():
    optimizer = Adam(learning_rate=BASE_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=CHANNELS, kernel_size=5, padding = 'same')])
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model
 
def record_number(input_list):
    total_sum = 0
    for string in input_list: 
        digit_list = findall(r'\d+', string)
        item_sum = sum([int(item) for item in digit_list])
        total_sum += item_sum
    return total_sum    

if __name__ == "__main__":   
       
    record_folder = 'F:/SR_folder/record_folder'
    train_list = glob.glob(record_folder + '/train_*.tfrecord')
    validation_list = glob.glob(record_folder + '/validation_*.tfrecord')
    
    train_data = _prepare_datasets(train_list)
    validation_data = _prepare_datasets(validation_list)
    
    validation_number = record_number(validation_list)
    train_number = record_number(train_list)
    
    model = create_model()    
    history = model.fit(train_data, epochs=EPOCHS, verbose=1, validation_data=validation_data,
                         steps_per_epoch=int(train_number/float(BATCH_SIZE)), validation_steps=int(validation_number/float(BATCH_SIZE)))
    model.save('result.h5')