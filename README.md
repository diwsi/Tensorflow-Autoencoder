# Tensorflow Autoencoder
A Tensorflow convolutional autoencoder sample with pretrained models. I trined my example to learn basic emboss effect samples.

# Architecture
Encoder model with  3 convolutional layers to extract features vector
```
Model: "EncoderModel" Input and 3 convolutional layers
_________________________________________________________________
Layer (type)                 Output Shape              Param #
_________________________________________________________________
InputLayer (InputLayer)      [(None, 256, 256, 3)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 256, 256, 16)      448
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 128, 128, 16)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 128, 128, 16)      2320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 64, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 16)        2320
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 32, 16)        0
_________________________________________________________________
Total params: 5,088
Trainable params: 5,088
Non-trainable params: 0
```

Decoder Model to reverse and upscale latent space 
```
_________________________________________________________________
Model: "DecoderModel"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
_________________________________________________________________
LatentSpace (InputLayer)     [(None, 32, 32, 16)]      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 16)        2320
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 64, 64, 16)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 64, 64, 16)        2320      
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 128, 128, 16)      0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 128, 128, 16)      2320
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 256, 256, 16)      0
_________________________________________________________________
OutputLayer (Conv2D)         (None, 256, 256, 3)       435
_________________________________________________________________
Total params: 7,395
Trainable params: 7,395
Non-trainable params: 0
```

Final autoencoder model that connects encoder and decoder
```
_________________________________________________________________
Model: "Autoencoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
_________________________________________________________________
InputLayer (InputLayer)      [(None, 256, 256, 3)]     0
_________________________________________________________________
EncoderModel (Model)         (None, 32, 32, 16)        5088
_________________________________________________________________
DecoderModel (Model)         (None, 256, 256, 3)       7395
_________________________________________________________________
Total params: 12,483
Trainable params: 12,483
Non-trainable params: 0
```
dd
