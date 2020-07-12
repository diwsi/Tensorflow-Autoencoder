import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import models, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import os
import cv2
import numpy as np
import time

def map_fn(x):
    x= tf.reshape(im,[-1, 256, 256, 3])    
    return x, x

files = tf.data.Dataset.list_files("img/x/*.*")

images = files.map(lambda f: tf.io.read_file(f))
images = images.map(lambda image: tf.io.decode_jpeg(image, channels=3))

images = images.map(lambda image: tf.cast(
    tf.image.resize(image, (256, 256)), tf.uint8))
images = images.map(lambda image: tf.cast(image, tf.float32) / 255)
trainData = images.map(map_fn)

print(trainData)


input = Input(shape=(256, 256, 3), name="L1")
for i in range(3):
    encLayer = Conv2D(16, (3, 3), padding='same',
                      activation='relu')(input if i == 0 else encLayer)
    encLayer = MaxPooling2D((2, 2), padding='same')(encLayer)

encoder = Model(inputs=input, outputs=encLayer)


encoded = Input(
    shape=(encLayer.shape[1], encLayer.shape[2], encLayer.shape[3]))
for i in range(3):
    decLayer = Conv2D(16, (3, 3), padding='same', activation='relu')(
        encoded if i == 0 else decLayer)
    decLayer = UpSampling2D((2, 2))(decLayer)
decLayer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decLayer)
decoder = Model(inputs=encoded, outputs=decLayer)


encDown = encoder(input)
encUp = decoder(encDown)
autoEncoder = Model(inputs=input, outputs=encUp)
autoEncoder.compile(optimizer='adam', loss='mse')
# print(autoEncoder.summary())

tbLog = TensorBoard(
    log_dir="logs\\gdmL{}".format(time.time()),
    histogram_freq=1)

history = autoEncoder.fit(trainData, batch_size=200,
                          epochs=10, callbacks=[tbLog])
decoder.save('decoder.h5')
encoder.save('encoder.h5')
autoEncoder.save('autoEncoder.h5')
