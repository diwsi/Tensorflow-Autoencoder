import numpy as np
import tensorflow as tf
import cv2
 

testData = []
#load model
model = tf.keras.models.load_model('autoEncoder.h5')
img = cv2.imread("./test/t.jpg")
img = cv2.resize(img, (256, 256))
cv2.imshow('Original', img) 

#apply actual filter
img = cv2.filter2D(img, -1, np.array([[0, -1, -1],
            [1, 0, -1],
            [1, 1, 0]]))
cv2.imshow('Target', img)             
cv2.imwrite("./test/tt.jpg", img)
#normalise and prepare data
img = img/255
testData.append(img)
testData = np.array(testData) 
testData = testData.reshape([-1, 256, 256, 3])

#prediction
res = model.predict(testData)

#reverse to data to picture
res = res*255
res = res.reshape(256, 256, 3).astype(np.uint8) 
cv2.imshow('Predicted', res)  
cv2.waitKey(0)
cv2.imwrite("./test/tp.jpg", res)
 