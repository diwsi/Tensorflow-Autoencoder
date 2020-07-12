import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import models, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import os
import cv2
import numpy as np
import time


paths = []
#load file paths
for subdir, dirs, files in os.walk("./img/x/"):
    for file in files:
        print(file)
        #append as (x,y)  tuple
        paths.append(("./img/x/"+file, "./img/y/"+file))

#Convert path array to tensors
files = tf.data.Dataset.from_tensor_slices(paths)
 

#Extract each (x,y) file in tensors and prepare for training
def extractFiles(path):
    #read actual and target files
    xh = tf.io.read_file(path[0])
    yh = tf.io.read_file(path[1])

    #decode jpeg images
    xh = tf.io.decode_jpeg(xh, channels=3)
    yh = tf.io.decode_jpeg(yh, channels=3)

    #resize to fixed 256x256 dimensions 
    xh = tf.cast(tf.image.resize(xh, (256, 256)), tf.uint8)
    yh = tf.cast(tf.image.resize(yh, (256, 256)), tf.uint8)

    #normalize data
    xh = tf.cast(xh, tf.float32) / 255
    yh = tf.cast(yh, tf.float32) / 255

    #reshape to desired input and output shapes
    xh = tf.reshape(xh, [-1, 256, 256, 3])
    yh = tf.reshape(yh, [-1, 256, 256, 3])
    return (xh, yh)
 

trainData=files.map(extractFiles)
  
#input layer for encoder
input=Input(shape=(256, 256, 3), name="InputLayer")

# 3 convolution downscale layer for encoder
for i in range(3):
    encLayer=Conv2D(16, (3, 3), padding='same',
                      activation='relu')(input if i == 0 else encLayer)
    encLayer=MaxPooling2D((2, 2), padding='same')(encLayer)

#Encoder Model
encoder=Model(inputs=input, outputs=encLayer,name="EncoderModel")
encoder.summary()
#latent space and input for decoder
encoded=Input(name="LatentSpace",
    shape=(encLayer.shape[1], encLayer.shape[2], encLayer.shape[3]))

# 3 convolution upscale layer for decoder
for i in range(3):
    decLayer=Conv2D(16, (3, 3), padding='same', activation='relu')(
        encoded if i == 0 else decLayer)
    decLayer=UpSampling2D((2, 2))(decLayer)

#final output layer    
decLayer=Conv2D(3, (3, 3),  activation='sigmoid', padding='same',name="OutputLayer" )(decLayer)

#decoder Model
decoder=Model(inputs=encoded, outputs=decLayer,name="DecoderModel")
decoder.summary()

#combine encoder and decoder models for autoencoder
encDown=encoder(input)
encUp=decoder(encDown)
autoEncoder=Model(inputs=input, outputs=encUp,name="Autoencoder")
autoEncoder.compile(optimizer='adam', loss='mse')
autoEncoder.summary()

#logs for tensorboard
tbLog=TensorBoard(
    log_dir="logs\\gdmL{}".format(time.time()),
    histogram_freq=1)

#train data
history=autoEncoder.fit(trainData, batch_size=200,
                          epochs=10, callbacks=[tbLog])

#Save models                                                                           
decoder.save('decoder.h5')
encoder.save('encoder.h5')
autoEncoder.save('autoEncoder.h5')
