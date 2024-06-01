import os
import random

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

#
# ##Create folder structure
#
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
#
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)
#
#
# cam = cv2.VideoCapture(0)
# while True:
#     ret_val, img = cam.read()
#     img = img[10:10 + 500, 900:900 + 500, :]
#
#     # collect anchors
#     if cv2.waitKey(1) & 0XFF == ord('a'):
#         imagename = os.path.join(ANC_PATH,'{}.jpg'.format((uuid.uuid1())))
#         cv2.imwrite(imagename,img)
#
#
#     #collect positive
#     if cv2.waitKey(1) & 0XFF == ord('p'):
#         imagename = os.path.join(POS_PATH, '{}.jpg'.format((uuid.uuid1())))
#         cv2.imwrite(imagename, img)
#
#     cv2.imshow('my webcam', img)
#     if cv2.waitKey(1) == 27:
#         break  # esc to quit
# cv2.destroyAllWindows()

#load and preprocess data

anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(50)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(50)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(50)
dir_test = anchor.as_numpy_iterator()
#
#
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

#
#create labelled dataset
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)  #creating a twin unlike a triplet loss

#
# # sample = data.as_numpy_iterator()
# print(sample.next())

def pre_process_twin(img, validation_img, label):
    return preprocess(img), preprocess(validation_img), label


#dataloader pipeline

data = data.map(pre_process_twin)  #apply the function to all samples
data = data.cache()
data = data.shuffle(buffer_size=1024)  #mixing up the data
#
# sample = data.as_numpy_iterator()
# samp = sample.next()
# print(samp[0])
#
# plt.imshow(samp[0])
# plt.show()
# print(f'This is length {len(samp)} and the label is {samp[2]}')
# plt.imshow(samp[1])
# plt.show()

#training partition

#training data partition
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

#testing partition

test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


#Model Engineering

def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = make_embedding()
print(embedding.summary())


#siamese L1 Distance class
# Siamese L1 Distance class
# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        subtraction = tf.math.subtract(input_embedding,validation_embedding)
        dist = tf.math.abs(subtraction)
        return tf.squeeze(dist,axis=0)

#Siamese model

def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
print(siamese_model.summary())
##PART 5

bcl = tf.losses.BinaryCrossentropy() #loss function

optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(opt = optimizer, siamese_model = siamese_model)


#training step for one batch of data
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape: #gradient tape helps to get the gradients

        X = batch[:2] #getting the anchor and positive/megative image

        y= batch[2] #getting the labels

        y_hat= siamese_model(X,training =True) #calculated value
        loss = bcl(y,y_hat)  #calculating loss

    #calculating gradeint
    grad = tape.gradient(loss,siamese_model.trainable_variables)

    #calculating updates weights a
    optimizer.apply_gradients(zip(grad,siamese_model.trainable_variables))


    return loss

#training loop
def train(data, epochs):
    #loop through all epochs
    for e in range(epochs):
        print(f'Epoch:{e}/{epochs}')
        progbar= tf.keras.utils.Progbar(len(data))
        #loop through each batch
        for i, batch in enumerate(data):
            train_step(batch)
            progbar.update(i+1)
        if e % 10==0:
            checkpoint.save(file_prefix=checkpoint_prefix)



#TRAIN model
EPOCH = 10

# train(train_data,epochs=EPOCH)


#Evaluate the model

test_input, test_val,y_true=test_data.as_numpy_iterator().next()
y_hat = siamese_model.predict([test_input,test_val])


print([1 if pred > 0.5 else 0 for pred in y_hat])

from tensorflow.keras.metrics import Precision, Recall
m = Precision()
m.update_state(y_true,y_hat)
m.result().numpy()

model = load_model('siamese_model.h5')
#verify the midel
def verify(model, detection_thresh, verification_threshold):
    #detection_threshold is the metric above which a prediction is considered positive
    #verification_threshold is the proportion positive predicitons/ total positive samples
    results = []

    for img in os.listdir(os.path.join('application_data','verification_images')):
        input_img = preprocess(os.path.join('application_data','input_img','input_img.jpg'))
        validation_img = preprocess(os.path.join('application_data','verification_images', img))

        result =  model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
        results.append(result)

    detection = np.sum(np.array(results)>detection_thresh)
    verification = detection / len(os.listdir(os.path.join('application_data','verification_images')))
    verified = verification > verification_threshold

    return results, verified


#real time verification

cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    img = img[10:10 + 500, 900:900 + 500, :]

    if cv2.waitKey(10) == 118:
       #save input image to input image folder
        cv2.imwrite(os.path.join('application_data','input_img','input_img.jpg'),img)
        results, verified = verify(model,0.5,0.5)
        print(f'This picture is verified? {verified}')
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()

